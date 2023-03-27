# we want batch sized to include multiples of typical number of cores as warp sizes
import multiprocessing


def get_batch_sizes():
    batches = set()
    batches.update(range(1, 16))
    batches.add(multiprocessing.cpu_count())
    batches.add(multiprocessing.cpu_count() // 2)

    for mult in range(36):
        for cores in [6, 8, 12, 16]:
            batches.add(cores * mult)
    batches = sorted(list(batches))
    return batches


if __name__ == "__main__":
    for i in get_batch_sizes():
        print(i)
