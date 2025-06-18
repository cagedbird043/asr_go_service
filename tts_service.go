// 文件: tts_service.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// WebSocket URL保持不变
const ttsWsURL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

// --- 数据结构 (保持不变) ---
type TTSHeader struct {
	Action    string `json:"action"`
	TaskID    string `json:"task_id"`
	Streaming string `json:"streaming"`
	Event     string `json:"event,omitempty"`
}

type TTSParameters struct {
	Voice      string `json:"voice"`
	Format     string `json:"format"`
	SampleRate int    `json:"sample_rate"`
}

type TTSInput struct {
	Text string `json:"text"`
}

type TTSPayload struct {
	TaskGroup  string         `json:"task_group,omitempty"`
	Task       string         `json:"task,omitempty"`
	Function   string         `json:"function,omitempty"`
	Model      string         `json:"model,omitempty"`
	Parameters *TTSParameters `json:"parameters,omitempty"`
	Input      *TTSInput      `json:"input,omitempty"`
}

type TTSRequest struct {
	Header  TTSHeader   `json:"header"`
	Payload *TTSPayload `json:"payload"`
}

type TTSResponseEvent struct {
	Header TTSHeader `json:"header"`
}

// --- 核心函数 (这是我们唯一需要对外暴露的函数) ---
func textToSpeech(textToSynthesize string) ([]byte, error) {
	apiKey := os.Getenv("DASHSCOPE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("DASHSCOPE_API_KEY environment variable not set")
	}

	header := http.Header{"Authorization": {"Bearer " + apiKey}}
	conn, _, err := websocket.DefaultDialer.Dial(ttsWsURL, header)
	if err != nil {
		return nil, fmt.Errorf("failed to dial TTS websocket: %v", err)
	}
	defer conn.Close()

	taskID := uuid.New().String()
	// ★★★ 核心修改：调整参数以匹配单片机的能力 ★★★
	runTaskCmd := TTSRequest{
		Header: TTSHeader{Action: "run-task", TaskID: taskID, Streaming: "duplex"},
		Payload: &TTSPayload{
			TaskGroup: "audio",
			Task:      "tts",
			Function:  "SpeechSynthesizer",
			Model:     "cosyvoice-v2",
			Parameters: &TTSParameters{
				Voice:      "longxiaochun_v2",
				Format:     "pcm", // 输出PCM，让单片机直接播放
				SampleRate: 16000, // 必须和单片机的I2S采样率一致
			},
			Input: &TTSInput{},
		},
	}
	if err := conn.WriteJSON(runTaskCmd); err != nil {
		return nil, fmt.Errorf("failed to send TTS run-task: %v", err)
	}

	// 等待 task-started
	var taskStarted bool
	for i := 0; i < 5; i++ {
		_, message, err := conn.ReadMessage()
		if err != nil {
			return nil, fmt.Errorf("error waiting for TTS task-started: %v", err)
		}
		var resp TTSResponseEvent
		if err := json.Unmarshal(message, &resp); err == nil && resp.Header.Event == "task-started" {
			taskStarted = true
			break
		}
	}
	if !taskStarted {
		return nil, fmt.Errorf("did not receive TTS task-started event")
	}

	// 发送文本
	continueCmd := TTSRequest{
		Header:  TTSHeader{Action: "continue-task", TaskID: taskID, Streaming: "duplex"},
		Payload: &TTSPayload{Input: &TTSInput{Text: textToSynthesize}},
	}
	if err := conn.WriteJSON(continueCmd); err != nil {
		return nil, fmt.Errorf("failed to send TTS continue-task: %v", err)
	}

	// 发送结束信号
	finishCmd := TTSRequest{
		Header:  TTSHeader{Action: "finish-task", TaskID: taskID, Streaming: "duplex"},
		Payload: &TTSPayload{Input: &TTSInput{}},
	}
	if err := conn.WriteJSON(finishCmd); err != nil {
		return nil, fmt.Errorf("failed to send TTS finish-task: %v", err)
	}

	// 接收所有音频数据
	var audioBuffer bytes.Buffer
	for {
		msgType, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
				break
			}
			return nil, fmt.Errorf("error reading TTS message: %v", err)
		}
		if msgType == websocket.BinaryMessage {
			audioBuffer.Write(message)
		} else if msgType == websocket.TextMessage {
			var resp TTSResponseEvent
			json.Unmarshal(message, &resp)
			if resp.Header.Event == "task-finished" {
				break
			}
			if resp.Header.Event == "task-failed" {
				return nil, fmt.Errorf("TTS task failed: %s", message)
			}
		}
	}

	return audioBuffer.Bytes(), nil
}

// ★★★ 核心修改：删除了原来的 main 函数 ★★★

// --------------------------------------------------------------------------------

// 	Main 函数 (用于独立测试)

// ----------------------------------------------------------------------
// func main() {
// 	log.Println("Starting TTS standalone test...")

// 	// 你可以修改这里的文本来测试不同的内容
// 	testText := "你好，世界。我是一个语音合成程序，正在测试我的发音是否标准。"

// 	log.Printf("Synthesizing text: \"%s\"\n", testText)

// 	// 调用核心逻辑
// 	audioData, err := textToSpeech(testText)
// 	if err != nil {
// 		log.Fatalf("textToSpeech failed: %v", err)
// 	}
// 	outputFile := "output_audio.pcm"
// 	// 将返回的音频数据写入文件
// 	err = os.WriteFile(outputFile, audioData, 0644)
// 	if err != nil {
// 		log.Fatalf("Failed to write to output file '%s': %v", outputFile, err)
// 	}

// 	log.Printf("Successfully synthesized audio and saved to %s (%d bytes).\n", outputFile, len(audioData))
// }
