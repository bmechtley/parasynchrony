There are different config files in the folder called marginals-x.json, all
  used for both slices and marginals code. with-sh-100.json is obsolete.

"file":
    "slice_size": chunks of parameter space for one process - literally number
        of parameter conbinations per process
"args":
    "resolution": resolution of the parameters (7^resolution calculations for 7
        parameters)
    "samplings": can include any number of entries formatted as follows:
        "XXXX": (any identifier)
            "range": range of fraction of synchrony to bin/sample according to these
                rules.
            "resolution": number of fraction of synchrony bins for this histogram
            "inclmin": whether or not to include the minimum value in the range
            "nsamples": number of random samples for which all params are stored.
    "params": default is for the slices only (marginals code and slices code both use
        this config file
        "param-symbol": one of r, a, c, k, mh, mp, SpSh, Chh, Cpp.
            default: default value, i.e. center of slices
            range: [low, high]

        Symbols are:
            r: host reproduction rate (non-trivial for > 1)
            a: host attack range
            c: no. parasitoid eggs laid per host.
            k: clumping parameter (stable for 0-1)
            mh: host migration factor (e.g. 0-0.5)
            mp: parasitoid migration factor
            SpSh: fraction of parasitoid variance to host variance (> 1 means
                parasitoid variance is higher)
            Chh: host-to-host correlation (0-1)
            Cpp: parasitoid-to-parasitoid correlation (0-1)