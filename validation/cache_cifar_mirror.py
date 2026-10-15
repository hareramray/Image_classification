"""One-time validation preparation; verify mirror bytes against torchvision's MD5."""
from pathlib import Path
import hashlib
import time
import requests

root = Path(__file__).resolve().parents[1]
target = root / "data" / "cifar-10-python.verified.tar.gz"
url = "https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/datasets/cifar/cifar-10-python.tar.gz"
digest = hashlib.md5()
start = time.perf_counter()
with requests.get(url, stream=True, timeout=(20, 60)) as response:
    response.raise_for_status()
    with target.open("wb") as output:
        for index, chunk in enumerate(response.iter_content(1024 * 1024)):
            output.write(chunk)
            digest.update(chunk)
            if index % 20 == 0:
                print(f"{output.tell() / 1e6:.1f} MB in {time.perf_counter() - start:.1f}s", flush=True)
assert digest.hexdigest() == "c58f30108f718f92721af3b95e74349a", "Dataset checksum mismatch"
print("Verified exact CIFAR-10 archive:", target, flush=True)
