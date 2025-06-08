from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
import json
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from models import get_db, create_tables, ChatHistory
from chatbot import TravelChatbot
from clustering import TravelRecommendationEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Travel Recommendation Chatbot started!")
    print("Loading recommendation engine...")
    yield
    # Shutdown
    print("Shutting down...")

# Khởi tạo FastAPI app
app = FastAPI(title="Travel Recommendation Chatbot", version="1.0.0", lifespan=lifespan)

# Setup templates và static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo các components
chatbot = TravelChatbot()
recommendation_engine = TravelRecommendationEngine()

# Tạo database tables
create_tables()
    
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(
    request: Request,
    db: Session = Depends(get_db)
):
    """API endpoint cho chat"""
    try:
        # Lấy dữ liệu từ request
        data = await request.json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", str(uuid.uuid4()))
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Trích xuất preferences từ user message
        print(f"Analyzing user message: {user_message}")
        preferences = chatbot.extract_travel_preferences(user_message)
        print(f"Extracted preferences: {preferences}")
        
        # Lấy recommendations
        recommendations = recommendation_engine.get_recommendations(preferences, top_k=5)
        print(f"Found {len(recommendations)} recommendations")
        
        # Tạo response
        bot_response = chatbot.generate_response(user_message, recommendations)
        
        # Lưu vào database
        chat_record = ChatHistory(
            user_message=user_message,
            bot_response=bot_response,
            extracted_features=json.dumps(preferences),
            recommended_locations=json.dumps(recommendations),
            user_ip=request.client.host,
            session_id=session_id
        )
        db.add(chat_record)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "response": bot_response,
            "recommendations": recommendations,
            "preferences": preferences,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "response": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."
        }, status_code=500)

@app.get("/api/clusters")
async def get_clusters():
    """API để lấy thông tin về các clusters"""
    try:
        clusters = recommendation_engine.get_all_clusters_summary()
        return JSONResponse({
            "success": True,
            "clusters": clusters
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/search/{location}")
async def search_location(location: str):
    """API để tìm kiếm theo tên địa điểm"""
    try:
        results = recommendation_engine.search_by_location(location, top_k=10)
        return JSONResponse({
            "success": True,
            "results": results,
            "total": len(results)
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/history")
async def get_chat_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """API để lấy lịch sử chat"""
    try:
        history = db.query(ChatHistory).order_by(
            ChatHistory.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for record in history:
            result.append({
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "user_message": record.user_message,
                "bot_response": record.bot_response,
                "session_id": record.session_id
            })
        
        return JSONResponse({
            "success": True,
            "history": result,
            "total": len(result)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

