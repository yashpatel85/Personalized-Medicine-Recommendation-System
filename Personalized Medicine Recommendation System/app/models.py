from pydantic import BaseModel, EmailStr
from typing import Optional

# For registration
class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: Optional[str] = "user"

# For login
class UserLogin(BaseModel):
    username: str
    password: str
