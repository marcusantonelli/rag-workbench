"""rag-crawl: a polite, focused crawler that collects on-topic documents up to a size budget.

It downloads only the file types the Builder can ingest, keeps files that are
actually about the topic (and in the chosen language), and stops near the
requested total size. robots.txt and crawl delays are obeyed, one request at a
time per host; nothing behind a login is fetched.
"""
