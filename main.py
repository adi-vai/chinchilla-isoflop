import torch


def main():
    data = [[1,2],[3,4]]
    tensor = torch.tensor(data)
    print("Hello from chinchilla-isoflop!")
    print(f"Here is our first tensor t: {tensor}")
    print(f"Running torch version {torch.__version__}")


if __name__ == "__main__":
    main()
