# Tool-mode `final_result` resume note

Tool-mode structured output currently ends without `resume_state` because the provider transcript contains an unanswered synthetic `final_result` tool call. That behavior is intentional today and documented as non-resumable.

One possible future design is to answer the synthetic tool call locally with a small success envelope after the structured output has been validated, then attach provider resume state from that completed transcript. That could make tool-mode structured output resumable for providers that require every tool call to receive a tool output before conversation state can continue.

This needs a design pass before implementation. The main questions are whether the synthetic tool output should be visible in raw provider responses, whether hooks/tracing should remain skipped for `final_result`, how native and prompted structured-output modes should be contrasted in docs, and which providers can safely resume after the synthetic acknowledgement.
