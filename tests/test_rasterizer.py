import torch

from humangs_profile.render import self_check


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    result = self_check()
    print(result)


if __name__ == "__main__":
    main()
