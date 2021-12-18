class ProcessConfig:
    def __init__(self, datapath, batch_size, device, hidden, n_res_blocks, n_res_subblocks, len_res_subblock, infer_path) -> None:
        self.datapath = datapath
        self.batch_size = batch_size
        self.device = device
        self.hidden = hidden
        self.n_res_blocks = n_res_blocks
        self.n_res_subblocks = n_res_subblocks
        self.len_res_subblock = len_res_subblock
        self.infer_path = infer_path
