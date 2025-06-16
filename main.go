package main

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// --- 配置常量 ---
const (
	aliyunWebSocketURL  = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
	sessionIdleTimeout  = 2 * time.Second
	pythonLLMServiceURL = "http://localhost:8000/chat" // 确保这是你Python后端的正确地址
)

// --- JSON 结构体定义 ---

// 用于调用Python后端的结构体
type ChatRequest struct {
	SessionID   string `json:"session_id"`
	UserMessage string `json:"user_message"`
}
type ChatResponse struct {
	AIResponse string `json:"ai_response"`
}

// 用于与阿里云WebSocket通信的结构体
type WsHeader struct {
	Action       string `json:"action"`
	TaskID       string `json:"task_id"`
	Streaming    string `json:"streaming,omitempty"`
	Event        string `json:"event,omitempty"`
	ErrorCode    string `json:"error_code,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
}
type WsInput struct{}
type WsPayload struct {
	TaskGroup  string       `json:"task_group,omitempty"`
	Task       string       `json:"task,omitempty"`
	Function   string       `json:"function,omitempty"`
	Model      string       `json:"model,omitempty"`
	Parameters WsParameters `json:"parameters,omitempty"`
	Input      WsInput      `json:"input"`
	Output     WsOutput     `json:"output,omitempty"`
}
type WsParameters struct {
	Format     string `json:"format"`
	SampleRate int    `json:"sample_rate"`
}
type WsOutput struct {
	Sentence struct {
		Text string `json:"text"`
	} `json:"sentence"`
}
type WsEvent struct {
	Header  WsHeader  `json:"header"`
	Payload WsPayload `json:"payload,omitempty"`
}

// --- 会话状态管理 ---
type ProcessingState int

const (
	StateIdle  ProcessingState = iota // 初始状态
	StateASR                          // 正在进行ASR
	StateLLM                          // 正在调用LLM
	StateDone                         // 所有处理完成
	StateError                        // 发生错误
)

type SessionState struct {
	SessionID        string
	apiKey           string
	pythonAPIKey     string
	pcmChannel       chan []byte
	InterimText      string
	FinalLLMResponse string
	Status           ProcessingState
	mu               sync.RWMutex
	stopOnce         sync.Once
}

var sessions = make(map[string]*SessionState)
var sessionsMu sync.Mutex

// --- `sessionWorker`: 核心工作goroutine ---
func sessionWorker(s *SessionState) {
	sessionID := s.SessionID
	s.mu.Lock()
	s.Status = StateASR
	s.mu.Unlock()
	log.Printf("[%s] Worker: 启动，状态 -> ASR\n", sessionID)

	defer func() {
		// 确保无论如何退出，会话状态都能被正确标记
		s.mu.Lock()
		if s.Status != StateDone {
			s.Status = StateError
			log.Printf("[%s] Worker: 异常退出，状态 -> Error\n", sessionID)
		}
		s.mu.Unlock()
	}()

	// 1. WebSocket 连接
	header := http.Header{"Authorization": {"Bearer " + s.apiKey}}
	conn, _, err := websocket.DefaultDialer.Dial(aliyunWebSocketURL, header)
	if err != nil {
		log.Printf("[%s] Worker: 连接失败: %v\n", sessionID, err)
		return
	}
	defer conn.Close()
	log.Printf("[%s] Worker: 连接成功。\n", sessionID)

	// 2. 启动消息接收goroutine
	taskStarted := make(chan bool, 1)
	taskDone := make(chan bool, 1)
	go func() {
		defer s.stopOnce.Do(func() { close(taskDone) })
		for {
			_, message, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var resp WsEvent
			if json.Unmarshal(message, &resp) != nil {
				continue
			}

			switch resp.Header.Event {
			case "task-started":
				taskStarted <- true
			case "result-generated":
				s.mu.Lock()
				s.InterimText = resp.Payload.Output.Sentence.Text
				s.mu.Unlock()
			case "task-finished":
				return
			case "task-failed":
				log.Printf("[%s] ASR任务失败: %s\n", sessionID, resp.Header.ErrorMessage)
				return
			}
		}
	}()

	// 3. 发送 run-task
	taskID := uuid.New().String()
	runTaskCmd := WsEvent{
		Header: WsHeader{Action: "run-task", TaskID: taskID, Streaming: "duplex"},
		Payload: WsPayload{
			TaskGroup: "audio", Task: "asr", Function: "recognition", Model: "paraformer-realtime-v2",
			Parameters: WsParameters{Format: "pcm", SampleRate: 16000},
			Input:      WsInput{},
		},
	}
	runTaskCmdJSON, _ := json.Marshal(runTaskCmd)
	if conn.WriteMessage(websocket.TextMessage, runTaskCmdJSON) != nil {
		return
	}

	select {
	case <-taskStarted:
		log.Printf("[%s] Worker: ASR任务开始。\n", sessionID)
	case <-time.After(10 * time.Second):
		log.Printf("[%s] Worker: 等待ASR任务开始超时。\n", sessionID)
		return
	}

	// 4. 创建持久化的ffmpeg进程
	ffmpegCmd := exec.Command("ffmpeg", "-f", "s16le", "-ar", "16000", "-ac", "2", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1")
	ffmpegStdin, _ := ffmpegCmd.StdinPipe()
	ffmpegStdout, _ := ffmpegCmd.StdoutPipe()
	if err := ffmpegCmd.Start(); err != nil {
		log.Printf("[%s] Worker: 启动ffmpeg失败: %v\n", sessionID, err)
		return
	}
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := ffmpegStdout.Read(buf)
			if err != nil {
				return
			}
			if conn.WriteMessage(websocket.BinaryMessage, buf[:n]) != nil {
				return
			}
		}
	}()

	// 5. 循环处理来自HTTP的PCM数据
	for {
		select {
		case stereoPcm, ok := <-s.pcmChannel:
			if !ok {
				goto finish
			}
			if _, err := ffmpegStdin.Write(stereoPcm); err != nil {
				goto finish
			}
		case <-time.After(sessionIdleTimeout):
			log.Printf("[%s] Worker: %v 内无新音频，超时。\n", sessionID, sessionIdleTimeout)
			goto finish
		}
	}

finish:
	ffmpegStdin.Close()
	ffmpegCmd.Wait()

	finishTaskCmd := WsEvent{Header: WsHeader{Action: "finish-task", TaskID: taskID}}
	finishTaskCmdJSON, _ := json.Marshal(finishTaskCmd)
	conn.WriteMessage(websocket.TextMessage, finishTaskCmdJSON)
	<-taskDone
	log.Printf("[%s] Worker: ASR流程结束。\n", sessionID)

	// 6. 调用Python大脑
	s.mu.RLock()
	finalASRText := s.InterimText
	s.mu.RUnlock()

	if finalASRText != "" {
		s.mu.Lock()
		s.Status = StateLLM
		s.mu.Unlock()
		log.Printf("[%s] Worker: 状态 -> LLM。调用Python大脑，文本: '%s'\n", sessionID, finalASRText)

		reqPayload := ChatRequest{SessionID: sessionID, UserMessage: finalASRText}
		reqBody, _ := json.Marshal(reqPayload)
		client := &http.Client{Timeout: 60 * time.Second}
		req, _ := http.NewRequest("POST", pythonLLMServiceURL, bytes.NewBuffer(reqBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-API-Key", s.pythonAPIKey)

		resp, err := client.Do(req)
		if err != nil {
			log.Printf("[%s] Worker: 调用Python失败: %v\n", sessionID, err)
		} else {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				var pyResponse ChatResponse
				if json.NewDecoder(resp.Body).Decode(&pyResponse) == nil {
					s.mu.Lock()
					s.FinalLLMResponse = pyResponse.AIResponse
					s.mu.Unlock()
					log.Printf("[%s] Worker: 收到Python回答: '%s'\n", sessionID, s.FinalLLMResponse)
				}
			} else {
				bodyBytes, _ := ioutil.ReadAll(resp.Body)
				log.Printf("[%s] Worker: Python返回错误 %d: %s\n", sessionID, resp.StatusCode, string(bodyBytes))
			}
		}
	}

	s.mu.Lock()
	s.Status = StateDone
	s.mu.Unlock()
	log.Printf("[%s] Worker: 状态 -> Done。任务全流程结束。\n", sessionID)
}

// --- Gin HTTP 处理器 ---
// 1. 用于流式上传音频
func asrStreamHandler(c *gin.Context) {
	sessionID := c.Param("session_id")
	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	pythonAPIKey := c.GetHeader("X-API-Key")

	sessionsMu.Lock()
	session, ok := sessions[sessionID]
	if !ok {
		log.Printf("[%s] HTTP-ASR: 新会话，创建worker。\n", sessionID)
		session = &SessionState{
			SessionID:    sessionID,
			apiKey:       apiKey,
			pythonAPIKey: pythonAPIKey,
			pcmChannel:   make(chan []byte, 100),
			Status:       StateIdle,
		}
		sessions[sessionID] = session
		go sessionWorker(session)
	}
	sessionsMu.Unlock()

	pcmData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无法读取请求体"})
		return
	}

	select {
	case session.pcmChannel <- pcmData:
	default:
		log.Printf("[%s] HTTP-ASR: pcmChannel已满或关闭。\n", sessionID)
	}

	session.mu.RLock()
	interimText := session.InterimText
	session.mu.RUnlock()
	c.JSON(http.StatusOK, gin.H{"interim_text": interimText})
}

// 2. 用于获取最终结果
func getResultHandler(c *gin.Context) {
	sessionID := c.Param("session_id")

	sessionsMu.Lock()
	session, ok := sessions[sessionID]
	sessionsMu.Unlock()

	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "session not found or expired"})
		return
	}

	session.mu.RLock()
	status := session.Status
	finalResponse := session.FinalLLMResponse
	session.mu.RUnlock()

	switch status {
	case StateDone:
		c.JSON(http.StatusOK, gin.H{"status": "completed", "response_text": finalResponse})
		// 获取后清理会话
		sessionsMu.Lock()
		delete(sessions, sessionID)
		sessionsMu.Unlock()
	case StateError:
		c.JSON(http.StatusOK, gin.H{"status": "error", "response_text": "An error occurred during processing."})
		sessionsMu.Lock()
		delete(sessions, sessionID)
		sessionsMu.Unlock()
	default: // Idle, ASR, LLM
		c.JSON(http.StatusOK, gin.H{"status": "processing"})
	}
}

// --- 主函数 ---
func main() {
	if os.Getenv("DASHSCOPE_API_KEY") == "" {
		log.Fatal("错误: DASHSCOPE_API_KEY 未设置！")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		log.Fatal("错误: ffmpeg 未找到！")
	}

	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	v1 := router.Group("/api/v1")
	{
		v1.POST("/asr/stream/:session_id", asrStreamHandler)
		v1.GET("/result/:session_id", getResultHandler)
	}

	gatewayPort := ":8080"
	log.Println("Go Unified Gateway (The Final Vision, Complete) is ready.")
	log.Println("Listening on: http://0.0.0.0" + gatewayPort)
	if err := router.Run(gatewayPort); err != nil {
		log.Fatalf("Gin服务启动失败: %v", err)
	}
}
