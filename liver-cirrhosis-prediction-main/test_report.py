import sys, warnings, asyncio, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'liver-cirrhosis-prediction-main', 'liver-cirrhosis-prediction-main', 'backend'))
os.chdir(os.path.join(os.path.dirname(__file__), 'liver-cirrhosis-prediction-main', 'liver-cirrhosis-prediction-main', 'backend'))

from database import SessionLocal, DiagnosisRecord, User, Base, engine
from passlib.context import CryptContext
Base.metadata.create_all(bind=engine)

db = SessionLocal()

pwd = CryptContext(schemes=['pbkdf2_sha256'], deprecated='auto')
u = db.query(User).filter(User.email == 'test@test.com').first()
if not u:
    u = User(email='test@test.com', hashed_password=pwd.hash('test1234'),
             full_name='Test User', role='patient')
    db.add(u); db.commit(); db.refresh(u)

user_id = u.id

d = db.query(DiagnosisRecord).filter(DiagnosisRecord.user_id == user_id).first()
if not d:
    d = DiagnosisRecord(
        user_id=user_id,
        symptoms={'age': 50, 'sex': 0, 'bilirubin': 2.5, 'albumin': 3.2},
        prediction_result={
            'stage': 'stage_3', 'stage_description': 'Stage 3 - Severe Fibrosis',
            'confidence': 0.82, 'risk_level': 'high',
            'probabilities': {'stage_0':0.05,'stage_1':0.08,'stage_2':0.10,'stage_3':0.77},
            'recommendations': ['Urgent hepatology review recommended.'],
            'follow_up_required': True, 'specialist_referral': True
        },
        explanations={
            'natural_language_explanation': 'AI predicts Stage 3 fibrosis.',
            'key_factors': ['elevated bilirubin', 'low albumin'],
            'confidence_interpretation': 'High confidence result.'
        },
        confidence_score=0.82, risk_level='high',
        recommended_actions=['Urgent hepatology review recommended.']
    )
    db.add(d); db.commit(); db.refresh(d)

diag_id = d.id
db.close()

from pdf_gen import MedicalReportGenerator
gen = MedicalReportGenerator()

async def run():
    result = await gen.generate_report(diag_id, user_id)
    print('PDF generated at:', result)
    print('File exists:', os.path.exists(result))
    print('File size:', os.path.getsize(result), 'bytes')

asyncio.run(run())
