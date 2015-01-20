import numpy as np
import matplotlib.pyplot as pp

import models


def main():
    model = models.parasitism.get_model("nbd(2)")
    nvars = len(model.vars)

    params = dict(r=3.0, a=0.5, c=1.2, k=0.9)
    params = {models.parasitism.symbols[k]: v for k, v in params.iteritems()}

    migrations = np.linspace(0, 1, 100)
    eigs = np.zeros((len(migrations), nvars), dtype=complex)

    print 'Computing eigenvalues.'
    for i, migration in enumerate(migrations):
        params[models.parasitism.symbols['mh']] = migration
        params[models.parasitism.symbols['mp']] = migration

        a = models.utilities.eval_matrix(model.A.subs(params))
        evals, evecs = np.linalg.eig(a)
        eigs[i] = evals

    print 'Plotting.'
    pp.figure()

    for i in range(nvars):
        pp.subplot(nvars, 2, i * 2 + 1)
        pp.plot(migrations, np.real(eigs[:, i]))
        pp.ylabel('$Re(\\lambda_%d)$' % (i + 1))
        pp.xlabel('$M_H = M_P$')

        pp.subplot(nvars, 2, i * 2 + 2)
        pp.plot(migrations, np.imag(eigs[:, i]))
        pp.ylabel('$Im(\\lambda_%d)$' % (i + 1))
        pp.xlabel('$M_H = M_P$')

        pp.subplots_adjust(wspace=0.5, hspace=0.5)

    pp.savefig('bifurcations.pdf')

    pp.figure(figsize=(4, 4))

    colors = [
        [(0.5 + f / 2, 0, 0) for f in migrations],
        [(0, 0.5 + f / 2, 0) for f in migrations],
        [(0, 0, 0.5 + f / 2) for f in migrations],
        [(0.5 + f / 2, 0.5 + f / 2, 0) for f in migrations]
    ]

    for i in range(nvars):
        for j in range(1, len(eigs) - 1):
            pp.plot(
                np.real(eigs[j-1:j+1, i]), np.imag(eigs[j-1:j+1, i]),
                color=colors[i][j],
                lw=5-i
            )

            # pp.scatter(
            #   np.real(eigs[:, i]),
            #   np.imag(eigs[:, i]),
            #   color=colors[i],
            #   s=(5-i)**2
            # )

    circlepts = [np.exp(1j * f) for f in np.linspace(0, 2 * np.pi, 100)]
    pp.plot(np.real(circlepts), np.imag(circlepts), color='k')

    pp.xlim(-1, 1)
    pp.ylim(-1, 1)
    pp.xlabel('$Re(\\lambda)$')
    pp.ylabel('$Im(\\lambda)$')

    pp.savefig('bifurcations-xy.pdf')


if __name__ == '__main__':
    main()