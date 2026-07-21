import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from tactus_align.io import load_score, synthesize_score
from config import SCORE_PATH

def test_synthesize_score():
    score = load_score(SCORE_PATH)
    Y, sr = synthesize_score(score)

    plt.plot(Y)
    plt.savefig("synthesized_score.png")

    return Y, sr

if __name__ == "__main__":
    Y, sr = test_synthesize_score()
    print(f"Y: {Y}, sr: {sr}")
    print(f"Y shape: {Y.shape}")