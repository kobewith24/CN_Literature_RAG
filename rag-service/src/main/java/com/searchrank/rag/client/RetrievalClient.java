package com.searchrank.rag.client;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import org.springframework.http.client.SimpleClientHttpRequestFactory;

/** 调用 Python 检索服务（BGE-M3 + HNSW + reranker）。 */
@Component
public class RetrievalClient {

    private final RestClient restClient;

    public RetrievalClient(@Value("${retrieval.base-url}") String baseUrl) {
        this.restClient = RestClient.builder().baseUrl(baseUrl).build();
    }

    public List<Hit> retrieve(String query, int k) {
        RetrieveResponse resp = restClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/retrieve")
                        .queryParam("q", query)
                        .queryParam("k", k)
                        .queryParam("mode", "hybrid")
                        .build())
                .retrieve()
                .body(RetrieveResponse.class);
        return resp == null || resp.hits() == null ? List.of() : resp.hits();
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RetrieveResponse(String query, String mode,
                                   @JsonProperty("elapsed_ms") Double elapsedMs,
                                   List<Hit> hits) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Hit(String source, Double score, String title, String text,
                      @JsonProperty("doc_id") String docId,
                      String category, String discipline, List<String> keywords,
                      @JsonProperty("rerank_score") Double rerankScore) {
    }
}
