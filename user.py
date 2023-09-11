from itsdangerous import TimedJSONWebSignatureSerializer as Serializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash

class User:
    secret_key = 'dfvbutNYUILh_kTDFBRJ!ors$t756yjkldfjh'

    def __init__(self, uid, name, password):
        self.id = uid
        self.name = name
        self.password_hash = generate_password_hash(password)

    def generate_auth_token(self, expiration=600):
        s = Serializer(User.secret_key, expires_in=expiration)
        return s.dumps({'id': self.id})

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def verify_auth_token(token, users):
        s = Serializer(User.secret_key)
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None# valid token, but expired
        except BadSignature:
            return None# invalid token

        uid = data['id']
        return next((u for u in users if u.id == uid), None)
