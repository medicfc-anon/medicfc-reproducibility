import re

def preprocess_claim(claim: str) -> str:
   
    claim = str(claim).lower().strip()
    claim = re.sub(r"\s+", " ", claim)
    return claim