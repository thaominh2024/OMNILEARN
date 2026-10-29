# app_story.py
import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")]

oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Story Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryIn(BaseModel):
    unit: str
    style: Optional[str] = "ngắn gọn, dí dỏm, 2-3 đoạn, có ví dụ minh họa"

class StoryOut(BaseModel):
    text: str

@app.post("/story", response_model=StoryOut)
def story(in_: StoryIn):
    if not OPENAI_API_KEY:
        txt = f"(FAKE) Truyện ngắn về '{in_.unit}' — (demo khi chưa có OPENAI_API_KEY)."
        return StoryOut(text=txt)
    sys = "Bạn viết truyện ngắn dễ hiểu để người mới cũng nắm được ý chính."
    user = f"Hãy viết truyện về: {in_.unit}\nPhong cách: {in_.style}\nNgôn ngữ: Tiếng Việt."
    rsp = oai.chat.completions.create(model=LLM_MODEL, temperature=0.7, messages=[
        {"role":"system","content":sys},{"role":"user","content":user}
    ])
    out = rsp.choices[0].message.content.strip()
    return StoryOut(text=out)

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8006, reload=True)
