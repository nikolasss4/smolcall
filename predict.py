"""Single prediction from the command line. Same as `smolcall predict`."""

from smolcall.cli import app

if __name__ == "__main__":
    app(["predict", *__import__("sys").argv[1:]])
