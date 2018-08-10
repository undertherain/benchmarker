from benchmarker.util.data.cubes import get_cubes


def get_data(params):
    return get_cubes(dims=2, edge=128, channels=1, cnt_samples=10*1024)


def main():
    data = get_data()
    print(data[0].shape)


if __name__ == "__main__":
    main()
