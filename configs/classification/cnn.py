import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        """
        #
        Returns:
            object:
        """
        super().__init__()

        self.add('--epoch', type=int, help='epoch number', default=500)
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-5])
        self.add('--b2', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.99])
        self.add('--b1', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.99])
        self.add('--mask', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.5])
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--model', help='Name of experiment', default="flipflop")
        self.add("--runs", action="store_true")
        self.add("--partial", action="store_true")
        self.add("--no-bootstrap", action="store_true")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--target-net', nargs='+', help='Seed', default=[10], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", nargs='+', type=int, default=[5])
        self.add("--sparsity", nargs='+', type=float, default=[0.5])
        self.add("--columns", nargs='+', type=int, default=[20])

