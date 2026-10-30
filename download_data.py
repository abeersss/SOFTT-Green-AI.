"""Download the unchanged public UCI archive and verify the benchmark checksum."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

URL='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
SHA256='b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4'
def main():
    dest=Path(__file__).resolve().parent/'data/beijing.zip'
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        payload=dest.read_bytes()
    else:
        with urlopen(URL,timeout=120) as response:payload=response.read()
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=SHA256:
        raise ValueError(f'Archive checksum mismatch: {digest}. Verify upstream version before use.')
    dest.write_bytes(payload)
    print(f'Verified {dest.name}: {len(payload):,} bytes')
if __name__=='__main__':main()
