import argparse

selection_parser = argparse.ArgumentParser(description='Global')

# -- Global --
selection_parser.add_argument('--index_create_duration', type=int, default=0,
                    help='time to build indexes')


def get_selection_args(argv=None):
    """Parse selection args only when explicitly requested."""
    return selection_parser.parse_args(argv)


# Default args (empty argv) so imports have usable defaults without CLI parsing
selection_args = get_selection_args([])


if __name__ == "__main__":
    print(get_selection_args())
