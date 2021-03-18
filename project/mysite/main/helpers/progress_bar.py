import sys


class ProgressBar:
    # Print iterations progress
    @staticmethod
    def progress_bar(iteration, total, prefix='', suffix='Complete', decimals=1, length=100, fill='█', printEnd="\r",
                     topbar=False):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        ProgressBar.print_progress_bar(iteration, decimals, fill, length, prefix, printEnd, suffix, total)

        # Print New Line on Complete
        if iteration == total:
            print()

    @staticmethod
    def print_progress_bar(iteration, decimals, fill, length, prefix, printEnd, suffix, total, topbar=False):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    @staticmethod
    def progress_bar_iterable(iterable, prefix='', suffix='Complete', decimals=1, length=100, fill='█', printEnd="\r", topbar=False):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)


        # Initial Call
        if topbar: ProgressBar.up()
        ProgressBar.print_progress_bar(0, decimals, fill, length, prefix, printEnd, suffix, total)
        # Update Progress Bar
        for i, item in enumerate(iterable):
            if topbar:
                ProgressBar.down()

            yield item

            if topbar:
                ProgressBar.up()
            ProgressBar.print_progress_bar(i + 1, decimals, fill, length, prefix, printEnd, suffix, total)

        # Print New Line on Complete
        print()

    @staticmethod
    def up():
        # My terminal breaks if we don't flush after the escape-code
        sys.stdout.write('\x1b[1A')
        sys.stdout.flush()

    @staticmethod
    def down():
        # I could use '\x1b[1B' here, but newline is faster and easier
        sys.stdout.write('\n')
        sys.stdout.flush()
