from random_knot import two_random_equivalent_knots
from visualize import visualize_knotified_braid
from matplotlib import pyplot as plt


def main():
    # Generate two random equivalent knots
    current_braid, target_braid = two_random_equivalent_knots(10, 20, 100)
    fig = visualize_knotified_braid(current_braid)
    plt.show()
    fig = visualize_knotified_braid(target_braid)
    plt.show()


if __name__ == '__main__':
    main()
