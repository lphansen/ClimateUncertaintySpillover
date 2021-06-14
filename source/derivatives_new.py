def dfunc_dx(data, dim, order, dlt, loc2, cap = None):
    """ compute the central difference derivatives for given input and dimensions;
    dim is the number of state varibles;
    order: order of the derivative;
    dlt: delta of variables"""

    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 4:
        if order == 1:                    # first order derivatives

            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / (2 * dlt)) * (data[2:,:,:,:] - data[:-2,:,:,:])
                res[-1,:,:,:] = (1 / dlt) * (data[-1,:,:,:] - data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt) * (data[1,:,:,:] - data[0,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / (2 * dlt)) * (data[:,2:,:,:] - data[:,:-2,:,:])
                res[:,-1,:,:] = (1 / dlt) * (data[:,-1,:,:] - data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt) * (data[:,1,:,:] - data[:,0,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / (2 * dlt)) * (data[:,:,2:,:] - data[:,:,:-2,:])
                res[:,:,-1,:] = (1 / dlt) * (data[:,:,-1,:] - data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt) * (data[:,:,1,:] - data[:,:,0,:])

            elif dim == 3:                # to forth dimension

                res[:,:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,:,2:] - data[:,:,:,:-2])
                res[:,:,:,-1] = (1 / dlt) * (data[:,:,:,-1] - data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt) * (data[:,:,:,1] - data[:,:,:,0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:

            if dim == 0:                  # to first dimension

                res[1:-1,:,:,:] = (1 / dlt ** 2) * (data[2:,:,:,:] + data[:-2,:,:,:] - 2 * data[1:-1,:,:,:])
                res[-1,:,:,:] = (1 / dlt ** 2) * (data[-1,:,:,:] + data[-3,:,:,:] - 2 * data[-2,:,:,:])
                res[0,:,:,:] = (1 / dlt ** 2) * (data[2,:,:,:] + data[0,:,:,:] - 2 * data[1,:,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:,:] = (1 / dlt ** 2) * (data[:,2:,:,:] + data[:,:-2,:,:] - 2 * data[:,1:-1,:,:])
                res[:,-1,:,:] = (1 / dlt ** 2) * (data[:,-1,:,:] + data[:,-3,:,:] - 2 * data[:,-2,:,:])
                res[:,0,:,:] = (1 / dlt ** 2) * (data[:,2,:,:] + data[:,0,:,:] - 2 * data[:,1,:,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1,:] = (1 / dlt ** 2) * (data[:,:,2:,:] + data[:,:,:-2,:] - 2 * data[:,:,1:-1,:])
                res[:,:,-1,:] = (1 / dlt ** 2) * (data[:,:,-1,:] + data[:,:,-3,:] - 2 * data[:,:,-2,:])
                res[:,:,0,:] = (1 / dlt ** 2) * (data[:,:,2,:] + data[:,:,0,:] - 2 * data[:,:,1,:])

            elif dim == 3:                # to third dimension

                res[:,:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,:,2:] + data[:,:,:,:-2] - 2 * data[:,:,:,1:-1])
                res[:,:,:,-1] = (1 / dlt ** 2) * (data[:,:,:,-1] + data[:,:,:,-3] - 2 * data[:,:,:,-2])
                res[:,:,:,0] = (1 / dlt ** 2) * (data[:,:,:,2] + data[:,:,:,0] - 2 * data[:,:,:,1])

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    elif l == 3:
        if order == 1:                    # first order derivatives

            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:

            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives

            if dim == 0:                  # to first dimension
                for i in range(loc1 +1):

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:

            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')

        else:
            raise ValueError('wrong order')
    else:
        if order == 1:                    # first order derivatives

            res[1:-1] = (1 / (2 * dlt)) * (data[2:] - data[:-2])
            res[-1] = (1 / dlt) * (data[-1] - data[-2])
            res[0] = (1 / dlt) * (data[1] - data[0])

        elif order == 2:

            res[1:-1] = (1 / dlt ** 2) * (data[2:] + data[:-2] - 2 * data[1:-1])
            res[-1] = (1 / dlt ** 2) * (data[-1] + data[-3] - 2 * data[-2])
            res[0] = (1 / dlt ** 2) * (data[2] + data[0] - 2 * data[1])
    if cap is not None:
        res[res < cap] = cap
    return res