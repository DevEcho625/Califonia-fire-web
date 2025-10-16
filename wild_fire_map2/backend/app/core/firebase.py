import firebase_admin
from firebase_admin import credentials, firestore
from app.core.config import settings

class FirebaseClient:
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": settings.FIREBASE_PROJECT_ID,
                "private_key_id": settings.FIREBASE_PRIVATE_KEY_ID,
                "private_key": settings.FIREBASE_PRIVATE_KEY.replace('\\n', '\n'),
                "client_email": settings.FIREBASE_CLIENT_EMAIL,
                "client_id": settings.FIREBASE_CLIENT_ID,
                "auth_uri": settings.FIREBASE_AUTH_URI,
                "token_uri": settings.FIREBASE_TOKEN_URI,
            })
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def get_collection(self, collection_name: str):
        return self.db.collection(collection_name)
    
    def get_document(self, collection_name: str, document_id: str):
        return self.db.collection(collection_name).document(document_id).get()

firebase_client = FirebaseClient()