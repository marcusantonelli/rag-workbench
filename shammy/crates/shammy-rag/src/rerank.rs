//! Reranking through an HTTP endpoint that speaks the `/v1/rerank` shape llama-server, Jina and
//! Cohere-compatible servers use:
//!
//!   POST {url}   {"model": ..., "query": ..., "documents": [...], "top_n": N}
//!   ->           {"results": [{"index": 0, "relevance_score": 0.93}, ...]}

use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

fn d_candidates() -> usize { 20 }
fn d_timeout() -> u64 { 15_000 }
fn d_max_chars() -> usize { 6000 }

#[derive(Clone, Debug, Deserialize)]
pub struct RerankOptions {
    #[serde(default)]
    pub enabled: bool,
    /// The rerank endpoint, e.g. "http://127.0.0.1:8081/v1/rerank". A bare "http://host:port" gets
    /// "/v1/rerank" added. Plain http only (a local model server).
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub model: String,
    /// How many fused candidates are reranked.
    #[serde(default = "d_candidates")]
    pub candidates: usize,
    #[serde(default = "d_timeout")]
    pub timeout_ms: u64,
    /// Passages are cut to this many characters before they are sent.
    #[serde(default = "d_max_chars")]
    pub max_chars: usize,
    #[serde(default)]
    pub api_key: String,
}

pub fn endpoint(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    let after_scheme = u.split_once("://").map_or(u, |p| p.1);
    if after_scheme.contains('/') { u.to_string() } else { format!("{u}/v1/rerank") }
}

fn cut(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// A relevance score for each document, in the order given. Err carries a message for the caller
/// to report; the search then keeps its fused order.
pub fn score(opts: &RerankOptions, query: &str, docs: &[&str]) -> Result<Vec<f64>, String> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    if opts.url.trim().is_empty() {
        return Err("reranking is on but no rerank URL is set".into());
    }
    let url = endpoint(&opts.url);
    if url.starts_with("https://") {
        return Err("the reranker must be a plain http endpoint (a local model server)".into());
    }
    let mut body = json!({"query": query, "top_n": docs.len(),
                          "documents": docs.iter().map(|d| cut(d, opts.max_chars)).collect::<Vec<_>>()});
    if !opts.model.is_empty() {
        body["model"] = json!(opts.model);
    }
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_millis(opts.timeout_ms.max(100))).build();
    let mut request = agent.post(&url).set("Content-Type", "application/json");
    if !opts.api_key.is_empty() {
        request = request.set("Authorization", &format!("Bearer {}", opts.api_key));
    }
    let response = request.send_string(&body.to_string()).map_err(|e| match e {
        ureq::Error::Status(code, r) => {
            let detail = r.into_string().unwrap_or_default();
            format!("the reranker at {url} answered {code}: {}", cut(detail.trim(), 300))
        }
        other => format!("can't reach the reranker at {url}: {other}"),
    })?;
    let text = response.into_string().map_err(|e| format!("the reranker's answer couldn't be read: {e}"))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("the reranker's answer isn't JSON: {e}"))?;
    let results = value.get("results").or_else(|| value.get("data")).and_then(Value::as_array)
        .ok_or_else(|| "the reranker's answer has no \"results\" list".to_string())?;
    let mut scores = vec![f64::NAN; docs.len()];
    for r in results {
        let index = r.get("index").and_then(Value::as_u64).map(|i| i as usize);
        let score = r.get("relevance_score").or_else(|| r.get("score")).and_then(Value::as_f64);
        if let (Some(i), Some(s)) = (index, score) {
            if i < scores.len() {
                scores[i] = s;
            }
        }
    }
    if scores.iter().any(|s| s.is_nan()) {
        return Err(format!("the reranker scored {} of {} passages", scores.iter().filter(|s| !s.is_nan()).count(), docs.len()));
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        assert_eq!(endpoint("http://127.0.0.1:8081"), "http://127.0.0.1:8081/v1/rerank");
        assert_eq!(endpoint("http://127.0.0.1:8081/"), "http://127.0.0.1:8081/v1/rerank");
        assert_eq!(endpoint("http://host:9/rerank"), "http://host:9/rerank");
        assert_eq!(cut("héllo", 2), "hé");
    }
}
