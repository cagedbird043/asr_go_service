package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// --- 配置常量 ---
const (
	mqttBroker          = "tcp://localhost:1883"
	pythonLLMServiceURL = "http://localhost:8000/chat"
	wsURL               = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
	cloudVadTimeout     = 5 * time.Second
)

// --- MQTT 主题 (Topics) ---
const (
	topicAudioStream   = "robot/audio/stream/+"
	topicControlPrefix = "robot/control/"
	topicResultPrefix  = "robot/result/"
)

// -- run-task 指令的数据结构 --
type ASRRequestHeader struct {
	Action    string `json:"action"`
	TaskID    string `json:"task_id"`
	Streaming string `json:"streaming"`
}

type ASRRequestParameters struct {
	Format                     string `json:"format"`
	SampleRate                 int    `json:"sample_rate"`
	SemanticPunctuationEnabled bool   `json:"semantic_punctuation_enabled"` // 开启云端智能断句
}

type ASRRequestPayload struct {
	TaskGroup  string                 `json:"task_group"`
	Task       string                 `json:"task"`
	Function   string                 `json:"function"`
	Model      string                 `json:"model"`
	Parameters ASRRequestParameters   `json:"parameters"`
	Input      map[string]interface{} `json:"input"`
}

type ASRRequestEvent struct {
	Header  ASRRequestHeader  `json:"header"`
	Payload ASRRequestPayload `json:"payload"`
}

// -- 服务端事件的数据结构 (完全参照文档重写) --
type ResponseHeader struct {
	TaskID       string `json:"task_id"`
	Event        string `json:"event"`
	ErrorCode    string `json:"error_code,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
}

type Sentence struct {
	BeginTime   int    `json:"begin_time"`
	EndTime     int    `json:"end_time"` // 在中间结果中为 null，Go的json库会自动处理
	Text        string `json:"text"`
	SentenceEnd bool   `json:"sentence_end"` // 这才是我们需要的、最关键的信号！
}

type Output struct {
	Sentence Sentence `json:"sentence"`
}

type ResponsePayload struct {
	Output Output `json:"output"`
}

type ResponseEvent struct {
	Header  ResponseHeader  `json:"header"`
	Payload ResponsePayload `json:"payload"`
}

// -- 其他辅助结构体 --
type ChatRequest struct {
	UserMessage string `json:"user_message"`
}

type ChatResponse struct {
	AIResponse string `json:"ai_response"`
}

// --- 会话状态管理 ---
type SessionState struct {
	SessionID    string
	pcmChannel   chan []byte
	stopChannel  chan bool
	finalASRText string
	mu           sync.RWMutex
}

var sessions = make(map[string]*SessionState)
var sessionsMu sync.Mutex
var mqttClient mqtt.Client

// logf: 一个带会话ID的日志打印函数
func logf(sessionID, format string, a ...interface{}) {
	log.Printf("[%s] %s", sessionID, fmt.Sprintf(format, a...))
}

// onMessageReceived: 所有MQTT消息的统一入口
func onMessageReceived(client mqtt.Client, msg mqtt.Message) {
	topic := msg.Topic()
	payload := msg.Payload()

	if strings.HasPrefix(topic, "robot/audio/stream/") {
		parts := strings.Split(topic, "/")
		if len(parts) != 4 {
			return
		}
		sessionID := parts[3]

		sessionsMu.Lock()
		session, ok := sessions[sessionID]
		if !ok {
			session = &SessionState{
				SessionID:   sessionID,
				pcmChannel:  make(chan []byte, 100),
				stopChannel: make(chan bool, 1),
			}
			sessions[sessionID] = session
			go sessionWorker(session)
			logf(sessionID, "New session started by first audio packet.")
		}
		sessionsMu.Unlock()

		select {
		case session.pcmChannel <- payload:
		default:
			logf(sessionID, "WARN: PCM channel is full. Discarding audio packet.")
		}
	}
}

const asrTurnTimeout = 2 * time.Second // 在最后一句话结束后，如果2秒没新话，就结束这一轮

// 1. ★★★★★★★★★★★★★★★★★ 用下面的代码，完整替换你现有的 sessionWorker 函数 ★★★★★★★★★★★★★★★★

func sessionWorker(s *SessionState) {
	sessionID := s.SessionID
	logf(sessionID, "Worker started for turn.")
	defer func() {
		sessionsMu.Lock()
		delete(sessions, sessionID)
		sessionsMu.Unlock()
		logf(sessionID, "Worker for turn finished and cleaned up.")
	}()

	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	header := http.Header{
		"Authorization": {"Bearer " + apiKey},
	}
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, header)
	if err != nil {
		logf(sessionID, "ERROR: Failed to connect to WebSocket: %v", err)
		return
	}
	defer conn.Close()
	logf(sessionID, "Connected to WebSocket.")

	taskStarted := make(chan bool, 1)
	var asrWg sync.WaitGroup
	asrWg.Add(1)

	doneChan := make(chan struct{})
	var closeOnce sync.Once // 这就是我们的“门卫”，确保门只被关一次

	// ASR 结果接收器 - 现在它完全基于云服务的语义断句信号
	go func() {
		defer asrWg.Done()

		turnTimer := time.NewTimer(asrTurnTimeout * 2)
		defer turnTimer.Stop()

		// 这个goroutine专门负责VAD超时
		go func() {
			<-turnTimer.C
			logf(sessionID, "ASR turn ended (no new speech for %v).", asrTurnTimeout)
			// 安全地关门
			closeOnce.Do(func() { close(doneChan) })
		}()

		var completeSentences []string
		var currentSentence string

		for {
			_, message, err := conn.ReadMessage()
			if err != nil {
				return
			}

			var resp ResponseEvent
			if err := json.Unmarshal(message, &resp); err != nil {
				continue
			}

			switch resp.Header.Event {
			case "task-started":
				taskStarted <- true

			case "result-generated":
				newText := resp.Payload.Output.Sentence.Text
				if newText == "" {
					continue
				}

				turnTimer.Reset(asrTurnTimeout)

				if resp.Payload.Output.Sentence.SentenceEnd {
					logf(sessionID, ">>> Received a complete sentence from cloud: '%s'", newText)
					completeSentences = append(completeSentences, newText)
					currentSentence = ""
				} else {
					currentSentence = newText
				}

				fullText := strings.Join(completeSentences, "") + currentSentence
				logf(sessionID, "ASR intermediate result: '%s'", fullText)

				s.mu.Lock()
				s.finalASRText = fullText
				s.mu.Unlock()

			case "task-finished", "task-failed":
				logf(sessionID, "ASR task ended (event: %s, msg: %s)", resp.Header.Event, resp.Header.ErrorMessage)
				// 安全地关门
				closeOnce.Do(func() { close(doneChan) })
				return
			}
		}
	}()

	taskID := uuid.New().String()
	runTaskCmd := ASRRequestEvent{
		Header: ASRRequestHeader{Action: "run-task", TaskID: taskID, Streaming: "duplex"},
		Payload: ASRRequestPayload{
			TaskGroup: "audio",
			Task:      "asr",
			Function:  "recognition",
			Model:     "paraformer-realtime-v2",
			Parameters: ASRRequestParameters{
				Format:                     "pcm",
				SampleRate:                 16000,
				SemanticPunctuationEnabled: true,
			},
			Input: make(map[string]interface{}),
		},
	}
	if err := conn.WriteJSON(runTaskCmd); err != nil {
		logf(sessionID, "ERROR: Failed to send run-task command: %v", err)
		return
	}

	select {
	case <-taskStarted:
		logf(sessionID, "ASR task started successfully. Now processing audio stream.")
	case <-time.After(10 * time.Second):
		logf(sessionID, "ERROR: Timeout waiting for task-started event.")
		return
	}

mainLoop:
	for {
		select {
		case pcmStereo, ok := <-s.pcmChannel:
			if !ok {
				break mainLoop
			}
			if len(pcmStereo)%4 != 0 {
				continue
			}
			numSamples := len(pcmStereo) / 4
			pcmMono := make([]byte, numSamples*2)
			for i := 0; i < numSamples; i++ {
				leftSample := int16(binary.LittleEndian.Uint16(pcmStereo[i*4 : i*4+2]))
				rightSample := int16(binary.LittleEndian.Uint16(pcmStereo[i*4+2 : i*4+4]))
				monoSample := int16((int32(leftSample) + int32(rightSample)) / 2)
				binary.LittleEndian.PutUint16(pcmMono[i*2:i*2+2], uint16(monoSample))
			}
			if err := conn.WriteMessage(websocket.BinaryMessage, pcmMono); err != nil {
				break mainLoop
			}
		case <-doneChan:
			break mainLoop
		}
	}

	logf(sessionID, "Audio processing finished.")

	time.Sleep(200 * time.Millisecond)

	finishTaskCmd := ASRRequestEvent{
		Header:  ASRRequestHeader{Action: "finish-task", TaskID: taskID, Streaming: "duplex"},
		Payload: ASRRequestPayload{Input: make(map[string]interface{})},
	}
	conn.WriteJSON(finishTaskCmd)

	asrWg.Wait()

	s.mu.RLock()
	finalText := s.finalASRText
	s.mu.RUnlock()

	if finalText == "" {
		logf(sessionID, "WARN: Final ASR text is empty. No result to publish.")
		return
	}

	logf(sessionID, "Final ASR Text: '%s'. Calling LLM...", finalText)

	llmReqPayload := map[string]string{"user_message": finalText}
	llmReqBody, _ := json.Marshal(llmReqPayload)
	llmClient := &http.Client{Timeout: 60 * time.Second}
	llmReq, _ := http.NewRequest("POST", pythonLLMServiceURL, bytes.NewBuffer(llmReqBody))
	llmReq.Header.Set("Content-Type", "application/json")
	llmReq.Header.Set("X-API-Key", "testkey1")

	llmResp, err := llmClient.Do(llmReq)
	if err != nil {
		logf(sessionID, "ERROR: Calling Python LLM failed: %v", err)
	} else {
		defer llmResp.Body.Close()
		var pyResp ChatResponse
		if llmResp.StatusCode == http.StatusOK && json.NewDecoder(llmResp.Body).Decode(&pyResp) == nil {
			logf(sessionID, "LLM response received: '%s'. Publishing to result topic.", pyResp.AIResponse)
			mqttClient.Publish(topicResultPrefix+sessionID, 0, false, []byte(pyResp.AIResponse))
		} else {
			bodyBytes, _ := io.ReadAll(llmResp.Body)
			logf(sessionID, "ERROR: LLM returned non-200 status (%d) or bad JSON. Body: %s", llmResp.StatusCode, string(bodyBytes))
		}
	}
}

// --- Main Function ---
func main() {
	if os.Getenv("DASHSCOPE_API_KEY") == "" {
		log.Fatal("ERROR: Environment variable DASHSCOPE_API_KEY is not set!")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		log.Fatal("ERROR: ffmpeg command not found in system PATH!")
	}

	opts := mqtt.NewClientOptions().AddBroker(mqttBroker).SetClientID("go_asr_gateway")
	opts.SetDefaultPublishHandler(onMessageReceived)

	mqttClient = mqtt.NewClient(opts)
	if token := mqttClient.Connect(); token.Wait() && token.Error() != nil {
		log.Fatalf("ERROR: Failed to connect to MQTT broker: %v", token.Error())
	}

	if token := mqttClient.Subscribe(topicAudioStream, 1, nil); token.Wait() && token.Error() != nil {
		log.Fatalf("ERROR: Failed to subscribe to topic '%s': %v", topicAudioStream, token.Error())
	}

	log.Println("Go MQTT Gateway is ready.")
	log.Printf("Connected to MQTT broker at %s", mqttBroker)
	log.Printf("Subscribed to audio topic: %s", topicAudioStream)

	select {}
}
