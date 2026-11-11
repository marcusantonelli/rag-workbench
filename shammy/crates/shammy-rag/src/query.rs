//! Turning a question into an FTS5 MATCH expression: identifiers as exact phrases, words OR-ed.
//! Everything here works on ASCII letters and digits only, as the Python reference does.

const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "i", "if",
    "in", "is", "it", "of", "on", "or", "our", "should", "that", "the", "their", "there", "these", "this", "to", "was",
    "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
];

/// `GetLotYield` -> get, lot, yield; `HTTPServer2` -> http, server, 2; `lot_id` -> lot, id.
pub fn split_identifier(name: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for piece in name.split('_') {
        let b = piece.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let run = |from: usize, test: fn(&u8) -> bool| from + b[from..].iter().take_while(|c| test(c)).count();
            let end = if b[i].is_ascii_uppercase() {
                let upper_end = run(i, u8::is_ascii_uppercase);
                let n = upper_end - i;
                let lower_next = upper_end < b.len() && b[upper_end].is_ascii_lowercase();
                if n >= 2 && lower_next {
                    upper_end - 1                           // "HTTP" of "HTTPServer"
                } else if n == 1 && lower_next {
                    run(upper_end, u8::is_ascii_lowercase)  // "Server"
                } else {
                    upper_end                               // "ID"
                }
            } else if b[i].is_ascii_lowercase() {
                run(i, u8::is_ascii_lowercase)
            } else if b[i].is_ascii_digit() {
                run(i, u8::is_ascii_digit)
            } else {
                i += 1;
                continue;
            };
            parts.push(piece[i..end].to_ascii_lowercase());
            i = end;
        }
    }
    parts
}

/// Tokens in a query that look like codes, part numbers or symbols.
pub fn query_identifiers(query: &str) -> Vec<String> {
    let b = query.as_bytes();
    let inner = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-' | b'/' | b'#');
    let mut found = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }
        // the longest run of token characters from here that also ends in a letter or digit
        let mut end = i + 1;
        while end < b.len() && inner(b[end]) {
            end += 1;
        }
        while end > i + 1 && !b[end - 1].is_ascii_alphanumeric() {
            end -= 1;
        }
        if end - i < 2 {
            i += 1; // one letter or digit alone is not a token; the scan moves on by one
            continue;
        }
        let tok = &query[i..end];
        let t = tok.as_bytes();
        let has_digit = t.iter().any(u8::is_ascii_digit);
        let has_sep = t.iter().any(|c| matches!(c, b'_' | b'-' | b'.' | b'/' | b'#'));
        let camel = t.windows(2).any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
        if (has_digit && tok.len() >= 3) || (has_sep && tok.len() >= 4) || camel {
            found.push(tok.to_string());
        }
        i = end;
    }
    found
}

pub fn fts_query(query: &str) -> String {
    let mut terms: Vec<String> = Vec::new();
    let push = |t: String, terms: &mut Vec<String>| {
        if !terms.contains(&t) {
            terms.push(t);
        }
    };
    for ident in query_identifiers(query) {
        push(format!("\"{}\"", ident.replace('"', "\"\"")), &mut terms);
        let parts = split_identifier(&ident);
        if parts.len() > 1 {
            push(format!("\"{}\"", parts.join(" ")), &mut terms);
        }
    }
    let b = query.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let w = query[start..i].to_ascii_lowercase();
        if w.len() >= 2 && !STOP.contains(&w.as_str()) {
            push(format!("\"{w}\""), &mut terms);
        }
    }
    terms.join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_split_like_the_reference() {
        let s = |x: &str| split_identifier(x).join(" ");
        assert_eq!(s("GetLotYield"), "get lot yield");
        assert_eq!(s("HTTPServer2"), "http server 2");
        assert_eq!(s("lot_id"), "lot id");
        assert_eq!(s("track-cli"), "track cli");
        assert_eq!(s("ABC"), "abc");
        assert_eq!(s("aB"), "a b");
        assert_eq!(s("XMLHttpRequest"), "xml http request");
    }

    #[test]
    fn query_tokens() {
        assert_eq!(query_identifiers("what does GetLotYield return for E42 in track-cli v2.1?"), ["GetLotYield", "E42", "track-cli", "v2.1"]);
        assert_eq!(query_identifiers("a b c"), Vec::<String>::new());
        assert_eq!(fts_query("How do I reset the E42 error?"), "\"E42\" OR \"e 42\" OR \"reset\" OR \"e42\" OR \"error\"");
        assert_eq!(fts_query("the of and"), "");
    }
}
