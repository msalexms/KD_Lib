from datasets import load_dataset
def example_usage():
    tiny_imagenet = load_dataset("zh-plus/tiny-imagenet", split='train')
    print(tiny_imagenet[0])

if __name__ == '__main__':
    example_usage()

