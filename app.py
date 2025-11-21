"""Flask web application for the LLM Q&A system."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from llm_qa_core import LLMClient, LLMQAService

app = Flask(__name__)
qa_service = LLMQAService(client=LLMClient())


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "question": "",
        "processed_question": "",
        "tokens": [],
        "answer": "",
        "error": "",
    }

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        context["question"] = question
        if not question:
            context["error"] = "Please enter a question."
        else:
            result = qa_service.answer_question(question)
            context.update(
                {
                    "processed_question": result["processed_question"],
                    "tokens": result["tokens"],
                    "answer": result["answer"],
                }
            )

    return render_template("index.html", **context)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    result = qa_service.answer_question(question)
    return jsonify(
        {
            "question": question,
            "processed_question": result["processed_question"],
            "tokens": result["tokens"],
            "answer": result["answer"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

