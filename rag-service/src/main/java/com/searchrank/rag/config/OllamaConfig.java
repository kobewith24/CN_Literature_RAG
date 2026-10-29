package com.searchrank.rag.config;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;

@Configuration
public class OllamaConfig {

    @Bean
    public ChatModel chatModel(@Value("${ollama.base-url}") String baseUrl,
                               @Value("${ollama.model}") String model) {
        return OllamaChatModel.builder()
                .baseUrl(baseUrl)
                .modelName(model)
                .temperature(0.3)
                .timeout(Duration.ofMinutes(3))
                .build();
    }

    @Bean
    public StreamingChatModel streamingChatModel(@Value("${ollama.base-url}") String baseUrl,
                                                 @Value("${ollama.model}") String model) {
        return OllamaStreamingChatModel.builder()
                .baseUrl(baseUrl)
                .modelName(model)
                .temperature(0.3)
                .timeout(Duration.ofMinutes(3))
                .build();
    }
}
