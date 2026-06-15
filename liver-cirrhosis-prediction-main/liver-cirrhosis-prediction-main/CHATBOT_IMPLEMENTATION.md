# Chatbot Implementation
## AI-Powered Healthcare Assistant

---

## 📊 Overview

This document details the implementation of an intelligent chatbot system for liver cirrhosis healthcare assistance. The chatbot combines NLP processing, symptom collection, medical Q&A, and LLM-based conversational AI.

---

## 🏗️ Chatbot Architecture

### 1. Core Components

```python
class LiverHealthChatbot:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.symptom_collector = SymptomCollector()
        self.medical_qa = MedicalQA()
        self.recommendation_engine = RecommendationEngine()
        self.conversation_manager = ConversationManager()
        self.llm_client = LLMClient()

    def process_message(self, user_message, session_context):
        """Main message processing pipeline"""
        # Step 1: Intent classification
        intent = self.nlp_processor.classify_intent(user_message)

        # Step 2: Entity extraction
        entities = self.nlp_processor.extract_entities(user_message)

        # Step 3: Context-aware response generation
        response = self.generate_response(intent, entities, session_context)

        return response
```

### 2. NLP Processor

```python
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.intent_classifier = self.load_intent_model()
        self.vectorizer = TfidfVectorizer()

    def classify_intent(self, message):
        """Classify user intent from message"""
        # Preprocess message
        processed_text = self.preprocess_text(message)

        # Vectorize
        vector = self.vectorizer.transform([processed_text])

        # Predict intent
        intent = self.intent_classifier.predict(vector)[0]

        return intent

    def extract_entities(self, message):
        """Extract medical entities from message"""
        doc = self.nlp(message)

        entities = {
            'symptoms': [],
            'medications': [],
            'conditions': [],
            'values': [],
            'dates': []
        }

        for ent in doc.ents:
            if ent.label_ in ['SYMPTOM', 'DISEASE']:
                entities['conditions'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)

        # Custom rule-based extraction for symptoms
        symptom_keywords = [
            'fatigue', 'tired', 'pain', 'swelling', 'jaundice',
            'nausea', 'vomiting', 'weight loss', 'appetite'
        ]

        for keyword in symptom_keywords:
            if keyword in message.lower():
                entities['symptoms'].append(keyword)

        return entities

    def preprocess_text(self, text):
        """Text preprocessing pipeline"""
        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])

        # Tokenize and lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]

        return ' '.join(tokens)
```

### 3. Symptom Collector

```python
class SymptomCollector:
    def __init__(self):
        self.symptom_questions = {
            'fatigue': "On a scale of 1-10, how severe is your fatigue?",
            'alcohol': "How many alcoholic drinks do you consume per week?",
            'weight_loss': "Have you experienced any weight loss recently? If yes, how much?",
            'abdominal_swelling': "Are you experiencing abdominal swelling or bloating?",
            'jaundice': "Have you noticed yellowing of your skin or eyes?",
            'appetite': "Have you lost your appetite or experienced nausea?",
            'pain': "Where do you experience pain? Please describe the intensity.",
            'fever': "Have you had any fever or elevated body temperature?"
        }

        self.collected_symptoms = {}

    def start_symptom_collection(self, session_id):
        """Initialize symptom collection session"""
        self.collected_symptoms[session_id] = {
            'current_question': 'fatigue',
            'answers': {},
            'completed': False
        }

        return {
            'message': "I'll help you assess your liver health. Let's start with some questions.",
            'next_question': self.symptom_questions['fatigue'],
            'question_type': 'fatigue'
        }

    def process_symptom_answer(self, session_id, question_type, answer):
        """Process user answer and move to next question"""
        if session_id not in self.collected_symptoms:
            return self.start_symptom_collection(session_id)

        session = self.collected_symptoms[session_id]
        session['answers'][question_type] = answer

        # Determine next question
        question_order = list(self.symptom_questions.keys())
        current_index = question_order.index(question_type)

        if current_index + 1 < len(question_order):
            next_question = question_order[current_index + 1]
            session['current_question'] = next_question

            return {
                'message': f"Thank you. {self.symptom_questions[next_question]}",
                'next_question': self.symptom_questions[next_question],
                'question_type': next_question,
                'progress': f"{current_index + 1}/{len(question_order)}"
            }
        else:
            # Collection complete
            session['completed'] = True
            return self.finalize_symptom_collection(session_id)

    def finalize_symptom_collection(self, session_id):
        """Complete symptom collection and prepare for prediction"""
        session = self.collected_symptoms[session_id]

        # Format symptoms for prediction
        formatted_symptoms = self.format_symptoms_for_prediction(session['answers'])

        return {
            'message': "Thank you for providing your symptoms. I'm analyzing your information now.",
            'symptoms_collected': True,
            'formatted_data': formatted_symptoms,
            'recommendation': "Would you like me to run a preliminary assessment based on your symptoms?"
        }

    def format_symptoms_for_prediction(self, answers):
        """Format collected symptoms for ML prediction"""
        formatted = {}

        # Map answers to prediction format
        mapping = {
            'fatigue': 'fatigue_level',
            'alcohol': 'alcohol_consumption',
            'weight_loss': 'weight_loss_kg',
            'abdominal_swelling': 'abdominal_swelling',
            'jaundice': 'jaundice',
            'appetite': 'appetite_loss'
        }

        for key, value in answers.items():
            if key in mapping:
                formatted[mapping[key]] = self.normalize_answer(key, value)

        return formatted

    def normalize_answer(self, question_type, answer):
        """Normalize user answers to numerical values"""
        if question_type == 'fatigue':
            # Convert scale answer to number
            try:
                return int(answer)
            except:
                return 5  # default

        elif question_type == 'alcohol':
            # Parse alcohol consumption
            try:
                return int(answer)
            except:
                return 0

        elif question_type in ['abdominal_swelling', 'jaundice', 'appetite']:
            # Convert yes/no to binary
            if answer.lower() in ['yes', 'y', 'true', '1']:
                return 1
            else:
                return 0

        return answer
```

### 4. Medical Q&A System

```python
class MedicalQA:
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.faq_embeddings = self.create_embeddings()

    def load_knowledge_base(self):
        """Load medical knowledge base"""
        return {
            'symptoms': {
                'jaundice': "Jaundice is yellowing of the skin and eyes caused by high bilirubin levels...",
                'fatigue': "Fatigue in liver disease can be caused by toxin buildup...",
                'abdominal_pain': "Abdominal pain may indicate liver inflammation or complications..."
            },
            'prevention': {
                'alcohol': "Limit alcohol consumption to prevent liver damage...",
                'diet': "Maintain a healthy diet rich in fruits, vegetables, and lean proteins..."
            },
            'treatment': {
                'early_stage': "Early stage cirrhosis may be managed with lifestyle changes...",
                'advanced_stage': "Advanced cirrhosis may require medical interventions..."
            }
        }

    def answer_question(self, question, context=None):
        """Answer medical questions using knowledge base and LLM"""
        # First, try knowledge base matching
        kb_answer = self.search_knowledge_base(question)

        if kb_answer:
            return {
                'answer': kb_answer,
                'source': 'knowledge_base',
                'confidence': 0.9
            }

        # Fallback to LLM
        llm_answer = self.llm_client.generate_medical_answer(question, context)

        return {
            'answer': llm_answer,
            'source': 'llm',
            'confidence': 0.7,
            'disclaimer': "This is general information. Please consult a healthcare professional."
        }

    def search_knowledge_base(self, question):
        """Search knowledge base for relevant information"""
        question_lower = question.lower()

        # Simple keyword matching
        for category, topics in self.knowledge_base.items():
            for topic, answer in topics.items():
                if topic in question_lower:
                    return answer

        return None
```

### 5. LLM Client

```python
import requests
import json

class LLMClient:
    def __init__(self, api_key=None, model="llama3-8b-8192"):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    def generate_medical_answer(self, question, context=None):
        """Generate medical answer using LLM"""
        system_prompt = """
        You are a helpful medical assistant specializing in liver health and cirrhosis.
        Provide accurate, helpful information about liver conditions, symptoms, and general health advice.
        Always include disclaimers that you're not a substitute for professional medical advice.
        Be empathetic and supportive in your responses.
        """

        user_prompt = f"""
        Question: {question}
        {f'Context: {context}' if context else ''}

        Please provide a helpful, accurate response about liver health.
        """

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return "I'm sorry, I'm having trouble accessing my knowledge base right now."

        except Exception as e:
            print(f"LLM API error: {e}")
            return "I'm experiencing technical difficulties. Please try again later."

    def generate_followup_questions(self, symptoms):
        """Generate relevant follow-up questions based on symptoms"""
        prompt = f"""
        Based on these symptoms: {symptoms}
        Generate 2-3 relevant follow-up questions a doctor might ask.
        Focus on liver health assessment.
        """

        # Similar API call structure
        # Return list of questions
```

### 6. Conversation Manager

```python
class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.max_session_length = 50  # messages

    def create_session(self, user_id):
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user_id,
            'messages': [],
            'context': {},
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }

        return session_id

    def add_message(self, session_id, message_type, content):
        """Add message to conversation history"""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Add message
        message = {
            'type': message_type,  # 'user' or 'bot'
            'content': content,
            'timestamp': datetime.now(),
            'message_id': len(session['messages'])
        }

        session['messages'].append(message)
        session['last_activity'] = datetime.now()

        # Maintain session length limit
        if len(session['messages']) > self.max_session_length:
            session['messages'] = session['messages'][-self.max_session_length:]

        return True

    def get_conversation_history(self, session_id, limit=10):
        """Get recent conversation history"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return session['messages'][-limit:]

    def update_context(self, session_id, key, value):
        """Update session context"""
        if session_id in self.sessions:
            self.sessions[session_id]['context'][key] = value

    def get_context(self, session_id):
        """Get session context"""
        if session_id in self.sessions:
            return self.sessions[session_id]['context']
        return {}

    def cleanup_old_sessions(self, max_age_hours=24):
        """Remove old inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session['last_activity'] < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]

        return len(sessions_to_remove)
```

### 7. Recommendation Engine

```python
class RecommendationEngine:
    def __init__(self):
        self.recommendations_db = self.load_recommendations()

    def load_recommendations(self):
        """Load recommendation templates"""
        return {
            'diet': {
                'low_sodium': "Reduce salt intake to less than 2g per day...",
                'high_protein': "Include lean proteins like chicken, fish...",
                'vegetables': "Eat plenty of fresh vegetables and fruits..."
            },
            'lifestyle': {
                'exercise': "Regular moderate exercise can help liver health...",
                'alcohol': "Complete abstinence from alcohol is recommended...",
                'smoking': "Quit smoking to improve overall health..."
            },
            'monitoring': {
                'follow_up': "Schedule regular check-ups with your doctor...",
                'symptoms': "Monitor for worsening symptoms and report them...",
                'medications': "Take prescribed medications regularly..."
            }
        }

    def generate_recommendations(self, prediction_result, symptoms):
        """Generate personalized recommendations"""
        recommendations = []

        risk_level = prediction_result.get('risk_level', 'low')
        stage = prediction_result.get('predicted_stage', 'stage_0')

        # Base recommendations
        recommendations.extend(self.get_base_recommendations(risk_level))

        # Symptom-specific recommendations
        recommendations.extend(self.get_symptom_recommendations(symptoms))

        # Stage-specific recommendations
        recommendations.extend(self.get_stage_recommendations(stage))

        return {
            'recommendations': recommendations,
            'priority': self.prioritize_recommendations(recommendations),
            'follow_up_schedule': self.generate_follow_up_schedule(risk_level)
        }

    def get_base_recommendations(self, risk_level):
        """Get basic recommendations based on risk level"""
        base_recs = []

        if risk_level in ['high', 'critical']:
            base_recs.extend([
                {
                    'category': 'urgent',
                    'title': 'Consult Hepatologist',
                    'description': 'Immediate consultation with liver specialist recommended',
                    'priority': 'high'
                },
                {
                    'category': 'lifestyle',
                    'title': 'Alcohol Cessation',
                    'description': self.recommendations_db['lifestyle']['alcohol'],
                    'priority': 'high'
                }
            ])

        base_recs.extend([
            {
                'category': 'diet',
                'title': 'Healthy Diet',
                'description': self.recommendations_db['diet']['vegetables'],
                'priority': 'medium'
            },
            {
                'category': 'monitoring',
                'title': 'Regular Monitoring',
                'description': self.recommendations_db['monitoring']['follow_up'],
                'priority': 'medium'
            }
        ])

        return base_recs

    def get_symptom_recommendations(self, symptoms):
        """Generate recommendations based on reported symptoms"""
        recs = []

        if symptoms.get('fatigue_level', 0) > 7:
            recs.append({
                'category': 'lifestyle',
                'title': 'Rest and Recovery',
                'description': 'Ensure adequate rest and avoid overexertion',
                'priority': 'medium'
            })

        if symptoms.get('jaundice', False):
            recs.append({
                'category': 'medical',
                'title': 'Jaundice Management',
                'description': 'Monitor jaundice progression and report changes',
                'priority': 'high'
            })

        return recs

    def get_stage_recommendations(self, stage):
        """Generate stage-specific recommendations"""
        recs = []

        if stage in ['stage_3', 'stage_4']:
            recs.extend([
                {
                    'category': 'medical',
                    'title': 'Advanced Care',
                    'description': 'May require specialized treatment or transplant evaluation',
                    'priority': 'high'
                },
                {
                    'category': 'monitoring',
                    'title': 'Close Monitoring',
                    'description': 'Frequent medical check-ups and monitoring required',
                    'priority': 'high'
                }
            ])

        return recs

    def prioritize_recommendations(self, recommendations):
        """Prioritize recommendations by urgency"""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}

        return sorted(recommendations,
                     key=lambda x: priority_order.get(x['priority'], 1),
                     reverse=True)

    def generate_follow_up_schedule(self, risk_level):
        """Generate recommended follow-up schedule"""
        schedules = {
            'low': [
                {'interval': '6 months', 'type': 'routine_check'},
                {'interval': 'annual', 'type': 'comprehensive'}
            ],
            'medium': [
                {'interval': '3 months', 'type': 'follow_up'},
                {'interval': '6 months', 'type': 'specialist'}
            ],
            'high': [
                {'interval': '1 month', 'type': 'urgent_follow_up'},
                {'interval': '3 months', 'type': 'specialist_consultation'}
            ],
            'critical': [
                {'interval': '2 weeks', 'type': 'immediate_care'},
                {'interval': '1 month', 'type': 'specialist_monitoring'}
            ]
        }

        return schedules.get(risk_level, schedules['medium'])
```

---

## 🔄 Chatbot Workflow

### 1. Message Processing Flow

```python
def process_user_message(user_id, message, session_id=None):
    """Main chatbot message processing function"""

    # Initialize or get session
    if not session_id:
        session_id = chatbot.conversation_manager.create_session(user_id)

    # Add user message to history
    chatbot.conversation_manager.add_message(session_id, 'user', message)

    # Process message
    response_data = chatbot.process_message(message, session_id)

    # Add bot response to history
    chatbot.conversation_manager.add_message(session_id, 'bot', response_data['message'])

    # Return response with session info
    return {
        'session_id': session_id,
        'response': response_data,
        'conversation_history': chatbot.conversation_manager.get_conversation_history(session_id)
    }
```

### 2. Intent Classification

```python
# Training data for intent classification
intent_training_data = [
    ("I'm feeling very tired", "symptom_reporting"),
    ("How can I prevent liver disease?", "health_education"),
    ("What are the symptoms of cirrhosis?", "medical_question"),
    ("I want to check my symptoms", "symptom_assessment"),
    ("Schedule a doctor's appointment", "appointment_booking"),
    ("What should I eat?", "diet_recommendation"),
    ("Tell me about my last test results", "result_inquiry"),
    ("I need help understanding my diagnosis", "diagnosis_explanation")
]

# Intent categories
INTENTS = {
    'symptom_reporting': 'User is reporting symptoms',
    'symptom_assessment': 'User wants to assess symptoms',
    'medical_question': 'User has medical question',
    'health_education': 'User wants health information',
    'diet_recommendation': 'User wants diet advice',
    'appointment_booking': 'User wants to schedule appointment',
    'result_inquiry': 'User wants test results',
    'diagnosis_explanation': 'User wants diagnosis explanation',
    'general_chat': 'General conversation'
}
```

### 3. Response Generation Strategy

```python
def generate_response(self, intent, entities, session_context):
    """Generate appropriate response based on intent and context"""

    response_strategies = {
        'symptom_reporting': self.handle_symptom_reporting,
        'symptom_assessment': self.handle_symptom_assessment,
        'medical_question': self.handle_medical_question,
        'health_education': self.handle_health_education,
        'diet_recommendation': self.handle_diet_recommendation,
        'appointment_booking': self.handle_appointment_booking,
        'result_inquiry': self.handle_result_inquiry,
        'diagnosis_explanation': self.handle_diagnosis_explanation,
        'general_chat': self.handle_general_chat
    }

    strategy = response_strategies.get(intent, self.handle_general_chat)
    return strategy(entities, session_context)
```

---

## 📊 Chatbot API Integration

### 1. FastAPI Chatbot Endpoints

```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.chatbot import ChatbotService

router = APIRouter()
chatbot_service = ChatbotService()

@router.post("/chat/start")
async def start_chat_session(user_id: int, db: Session = Depends(get_db)):
    """Start a new chat session"""
    try:
        session_id = chatbot_service.create_session(user_id)

        welcome_message = {
            "message": "Hello! I'm your liver health assistant. How can I help you today?",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        return {"success": True, "data": welcome_message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")

@router.post("/chat/message")
async def send_message(
    session_id: str,
    message: str,
    user_id: int = None,
    db: Session = Depends(get_db)
):
    """Send message to chatbot"""
    try:
        response = chatbot_service.process_message(session_id, message, user_id)

        return {
            "success": True,
            "data": {
                "response": response["message"],
                "intent": response.get("intent"),
                "entities": response.get("entities", {}),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 20):
    """Get chat history for session"""
    try:
        history = chatbot_service.get_conversation_history(session_id, limit)

        return {"success": True, "data": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/chat/symptom-assessment/{session_id}")
async def start_symptom_assessment(session_id: str):
    """Start symptom assessment workflow"""
    try:
        assessment = chatbot_service.start_symptom_assessment(session_id)

        return {"success": True, "data": assessment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")
```

### 2. React Frontend Integration

```javascript
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

function Chatbot({ userId }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    // Start chat session on component mount
    startChatSession();
  }, []);

  const startChatSession = async () => {
    try {
      const response = await axios.post('/api/chat/start', { user_id: userId });
      setSessionId(response.data.data.session_id);
      setMessages([{
        type: 'bot',
        content: response.data.data.message,
        timestamp: response.data.data.timestamp
      }]);
    } catch (error) {
      console.error('Failed to start chat:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !sessionId) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await axios.post('/api/chat/message', {
        session_id: sessionId,
        message: inputMessage,
        user_id: userId
      });

      const botMessage = {
        type: 'bot',
        content: response.data.data.response,
        intent: response.data.data.intent,
        timestamp: response.data.data.timestamp
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <h3>Liver Health Assistant</h3>
        <span className="status">Online</span>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            <div className="message-content">
              {message.content}
            </div>
            <div className="message-time">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="message bot typing">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isTyping}
        />
        <button
          onClick={sendMessage}
          disabled={!inputMessage.trim() || isTyping}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default Chatbot;
```

---

## 🎯 Chatbot Features Implementation

### 1. Symptom Collection Workflow

```python
def handle_symptom_assessment(self, entities, session_context):
    """Handle symptom assessment requests"""
    session_id = session_context.get('session_id')

    # Check if assessment already in progress
    if session_context.get('assessment_active'):
        return self.continue_symptom_assessment(session_id, entities)

    # Start new assessment
    assessment_data = self.symptom_collector.start_symptom_collection(session_id)

    # Update session context
    self.conversation_manager.update_context(session_id, 'assessment_active', True)
    self.conversation_manager.update_context(session_id, 'current_question', 'fatigue')

    return {
        'message': assessment_data['message'],
        'next_question': assessment_data['next_question'],
        'question_type': assessment_data['question_type'],
        'assessment_started': True
    }

def continue_symptom_assessment(self, session_id, entities):
    """Continue ongoing symptom assessment"""
    current_question = self.conversation_manager.get_context(session_id).get('current_question')

    # Process answer
    result = self.symptom_collector.process_symptom_answer(
        session_id, current_question, entities.get('answer', '')
    )

    if result.get('symptoms_collected'):
        # Assessment complete
        self.conversation_manager.update_context(session_id, 'assessment_active', False)
        self.conversation_manager.update_context(session_id, 'symptoms_data', result['formatted_data'])

        return {
            'message': result['message'],
            'assessment_complete': True,
            'symptoms_data': result['formatted_data'],
            'recommendation': result['recommendation']
        }
    else:
        # Continue assessment
        self.conversation_manager.update_context(session_id, 'current_question', result['question_type'])

        return {
            'message': result['message'],
            'next_question': result['next_question'],
            'question_type': result['question_type'],
            'progress': result.get('progress')
        }
```

### 2. Medical Q&A Implementation

```python
def handle_medical_question(self, entities, session_context):
    """Handle medical questions"""
    question = entities.get('question', '')

    # Try knowledge base first
    kb_answer = self.medical_qa.search_knowledge_base(question)

    if kb_answer:
        return {
            'message': kb_answer,
            'source': 'knowledge_base',
            'confidence': 'high'
        }

    # Use LLM for complex questions
    context = self.build_medical_context(session_context)
    llm_answer = self.llm_client.generate_medical_answer(question, context)

    return {
        'message': llm_answer,
        'source': 'ai_assistant',
        'disclaimer': 'This is general information. Please consult a healthcare professional for personalized advice.',
        'confidence': 'medium'
    }

def build_medical_context(self, session_context):
    """Build context for medical questions"""
    context = ""

    # Add user symptoms if available
    symptoms_data = session_context.get('symptoms_data', {})
    if symptoms_data:
        context += f"User symptoms: {symptoms_data}\n"

    # Add recent predictions if available
    recent_prediction = session_context.get('recent_prediction')
    if recent_prediction:
        context += f"Recent prediction: {recent_prediction}\n"

    return context
```

### 3. Recommendation Generation

```python
def handle_diet_recommendation(self, entities, session_context):
    """Handle diet recommendation requests"""
    symptoms_data = session_context.get('symptoms_data', {})
    prediction_data = session_context.get('recent_prediction', {})

    recommendations = self.recommendation_engine.generate_recommendations(
        prediction_data, symptoms_data
    )

    # Format response
    response = "Based on your health profile, here are some dietary recommendations:\n\n"

    for rec in recommendations['recommendations'][:3]:  # Top 3
        response += f"• {rec['title']}: {rec['description'][:100]}...\n"

    if recommendations.get('follow_up_schedule'):
        response += f"\nRecommended follow-up: {recommendations['follow_up_schedule'][0]['interval']}"

    return {
        'message': response,
        'recommendations': recommendations['recommendations'],
        'follow_up': recommendations.get('follow_up_schedule'),
        'category': 'diet'
    }
```

---

## 🔧 Configuration and Setup

### 1. Environment Variables

```bash
# Chatbot Configuration
GROQ_API_KEY=your_groq_api_key_here
CHATBOT_MODEL=llama3-8b-8192
MAX_SESSION_LENGTH=50
SESSION_TIMEOUT_HOURS=24

# NLP Configuration
SPACY_MODEL=en_core_web_sm
INTENT_MODEL_PATH=models/intent_classifier.pkl
VECTORIZER_PATH=models/tfidf_vectorizer.pkl
```

### 2. Dependencies

```python
# requirements_chatbot.txt
spacy==3.7.2
scikit-learn==1.3.0
nltk==3.8.1
requests==2.31.0
transformers==4.30.0
torch==2.0.1
pandas==2.0.3
numpy==1.24.3
python-dotenv==1.0.0
```

### 3. Model Training

```python
def train_intent_classifier():
    """Train intent classification model"""
    # Load training data
    X_train, y_train = load_intent_training_data()

    # Create pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Save model
    import joblib
    joblib.dump(pipeline, 'models/intent_classifier.pkl')

    return pipeline
```

---

## 📊 Performance Monitoring

### 1. Chatbot Analytics

```python
class ChatbotAnalytics:
    def __init__(self):
        self.metrics = {
            'total_sessions': 0,
            'total_messages': 0,
            'intent_distribution': {},
            'response_times': [],
            'user_satisfaction': []
        }

    def log_interaction(self, session_id, intent, response_time, user_feedback=None):
        """Log chatbot interaction"""
        self.metrics['total_messages'] += 1

        # Intent distribution
        self.metrics['intent_distribution'][intent] = \
            self.metrics['intent_distribution'].get(intent, 0) + 1

        # Response time
        self.metrics['response_times'].append(response_time)

        # User feedback
        if user_feedback:
            self.metrics['user_satisfaction'].append(user_feedback)

    def get_performance_report(self):
        """Generate performance report"""
        return {
            'total_sessions': self.metrics['total_sessions'],
            'total_messages': self.metrics['total_messages'],
            'avg_response_time': np.mean(self.metrics['response_times']),
            'intent_breakdown': self.metrics['intent_distribution'],
            'user_satisfaction_score': np.mean(self.metrics['user_satisfaction']) if self.metrics['user_satisfaction'] else None
        }
```

---

**For implementation details, see** `backend/services/chatbot_service.py` and `backend/utils/nlp_processor.py`
