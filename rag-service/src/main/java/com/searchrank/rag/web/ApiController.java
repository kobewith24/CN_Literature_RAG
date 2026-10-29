package com.searchrank.rag.web;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.searchrank.rag.client.RetrievalClient;
import com.searchrank.rag.client.RetrievalClient.Hit;
import com.searchrank.rag.service.RagService;

@RestController
@RequestMapping("/api")
public class ApiController {

    private final RagService ragService;
    private final RetrievalClient retrievalClient;

    public ApiController(RagService ragService, RetrievalClient retrievalClient) {
        this.ragService = ragService;
        this.retrievalClient = retrievalClient;
    }

    /** 流式问答（SSE）：sources -> token* -> done */
    @GetMapping("/ask")
    public SseEmitter ask(@RequestParam("q") String q,
                          @RequestParam(value = "k", defaultValue = "5") int k) {
        SseEmitter emitter = new SseEmitter(300_000L);
        ragService.askStream(q, k, emitter);
        return emitter;
    }

    /** 纯检索（不生成答案） */
    @GetMapping("/search")
    public List<Hit> search(@RequestParam("q") String q,
                            @RequestParam(value = "k", defaultValue = "10") int k) {
        return retrievalClient.retrieve(q, k);
    }
}
