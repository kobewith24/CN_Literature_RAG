package com.searchrank.rag.service;

import java.io.IOException;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.searchrank.rag.client.RetrievalClient;
import com.searchrank.rag.client.RetrievalClient.Hit;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

/** 组装检索结果 + Prompt，调用 Ollama 流式生成答案。 */
@Service
public class RagService {

    private final RetrievalClient retrievalClient;
    private final StreamingChatModel streamingChatModel;
    private final ObjectMapper objectMapper;

    public RagService(RetrievalClient retrievalClient,
                      StreamingChatModel streamingChatModel,
                      ObjectMapper objectMapper) {
        this.retrievalClient = retrievalClient;
        this.streamingChatModel = streamingChatModel;
        this.objectMapper = objectMapper;
    }

    private static String nz(String s) {
        return s == null ? "" : s;
    }

    public String buildPrompt(String question, List<Hit> hits) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是文献检索与问答助手。请严格依据下面【资料】回答用户问题，不要编造资料以外的内容；")
          .append("资料不足以回答时请直接说明。回答简洁，引用处用 [编号] 标注。\n\n【资料】\n");
        int i = 1;
        for (Hit h : hits) {
            sb.append("[").append(i++).append("] ").append(nz(h.title())).append("\n");
            if (h.text() != null && !h.text().isBlank()) {
                sb.append("摘要：").append(h.text()).append("\n");
            }
            if (h.discipline() != null && !h.discipline().isBlank()) {
                sb.append("学科：").append(h.discipline()).append("\n");
            }
            sb.append("\n");
        }
        sb.append("【问题】\n").append(question).append("\n");
        return sb.toString();
    }

    public void askStream(String question, int k, SseEmitter emitter) {
        try {
            List<Hit> hits = retrievalClient.retrieve(question, k);
            emitter.send(SseEmitter.event().name("sources").data(objectMapper.writeValueAsString(hits)));

            String prompt = buildPrompt(question, hits);
            streamingChatModel.chat(prompt, new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String partialResponse) {
                    try {
                        emitter.send(SseEmitter.event().name("token").data(partialResponse));
                    } catch (IOException ignored) {
                        // 客户端已断开
                    }
                }

                @Override
                public void onCompleteResponse(ChatResponse completeResponse) {
                    try {
                        emitter.send(SseEmitter.event().name("done").data(""));
                    } catch (IOException ignored) {
                        // 忽略
                    }
                    emitter.complete();
                }

                @Override
                public void onError(Throwable error) {
                    try {
                        emitter.send(SseEmitter.event().name("error").data(String.valueOf(error.getMessage())));
                    } catch (IOException ignored) {
                        // 忽略
                    }
                    emitter.completeWithError(error);
                }
            });
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
    }
}
