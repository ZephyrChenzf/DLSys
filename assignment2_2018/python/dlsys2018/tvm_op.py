from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = te.placeholder(shape, dtype=dtype, name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = te.create_schedule(C.op)
    if tgt=="cuda":
        bx,tx=s[C].split(C.op.axis[0],factor=64)
        s[C].bind(bx,te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A=te.placeholder(shape,dtype = dtype, name="A")
    B=te.placeholder(shape,dtype = dtype, name="B")
    C=te.compute(A.shape,lambda *i:A(*i)*B(*i))
    s=te.create_schedule(C.op)
    if tgt=="cuda":
        bx,tx=s[C].split(C.op.axis[0],factor=64)
        s[C].bind(bx,te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

    f=tvm.build(s,[A,B,C],tgt,tgt_host,name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A=te.placeholder(shape,dtype=dtype,name="A")
    B=te.compute(A.shape,lambda *i: A(*i)+const_k)
    s=te.create_schedule(B.op)
    if tgt=="cuda":
        bx,tx=s[B].split(B.op.axis[0],factor=64)
        s[B].bind(bx,te.thread_axis("blockIdx.x"))
        s[B].bind(tx, te.thread_axis("threadIdx.x"))

    f=tvm.build(s,[A,B],tgt,target_host=tgt_host,name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A=te.placeholder(shape,dtype=dtype, name="A")
    B=te.compute(A.shape,lambda *i:A[i]*const_k)
    s=te.create_schedule(B.op)
    if tgt=="cuda":
        bx,tx=s[B].split(B.op.axis[0],factor=64)
        s[B].bind(bx,te.thread_axis("blockIdx.x"))
        s[B].bind(tx, te.thread_axis("threadIdx.x"))

    f=tvm.build(s,[A,B],tgt,tgt_host,name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A=te.placeholder(shape,dtype = dtype, name="A")
    B=te.compute(A.shape,lambda *i:te.if_then_else(A(*i)>0,A(*i),0))
    s=te.create_schedule(B.op)
    if tgt=="cuda":
        bx,tx=s[B].split(B.op.axis[0],factor=64)
        s[B].bind(bx,te.thread_axis("blockIdx.x"))
        s[B].bind(tx, te.thread_axis("threadIdx.x"))

    f=tvm.build(s,[A,B],tgt,tgt_host,name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A=te.placeholder(shape,dtype=dtype,name="A")
    B=te.placeholder(shape,dtype = dtype, name="B")
    C=te.compute(A.shape,lambda *i:te.if_then_else(A(*i)>0,B(*i),0))
    s=te.create_schedule(C.op)
    if tgt=="cuda":
        bx,tx=s[C].split(C.op.axis[0],factor=64)
        s[C].bind(bx,te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

    f=tvm.build(s,[A,B,C],tgt,tgt_host,name=func_name)
    return f



def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A=te.placeholder(shapeA,dtype=dtype,name="A")
    B=te.placeholder(shapeB,dtype=dtype, name="B")
    def transpose(mat):
        return te.compute((mat.shape[1],mat.shape[0]),lambda i,j:mat[j][i])


    AA=A if not transposeA else transpose(A)
    BB=B if not transposeB else transpose(B)
    k=te.reduce_axis((0,AA.shape[1]),name="k")
    C=te.compute((AA.shape[0],BB.shape[1]),lambda i,j:te.sum(AA[i][k]*BB[k][j],axis =k))

    s=te.create_schedule(C.op)
    if tgt=="llvm":
        xo,yo,xi,yi=s[C].tile(C.op.axis[0],C.op.axis[1],32,32)
        k,=s[C].op.reduce_axis
        ko,ki=s[C].split(k,factor=4)
        s[C].reorder(xo,yo,ko,xi,yi,ki)
        # s[C].parallel(ki)
    if tgt=="cuda":
        if transposeA:
            xx1,xx2=s[AA].split(AA.op.axis[0],factor=32)
            s[AA].bind(xx1,te.thread_axis("blockIdx.x"))
            s[AA].bind(xx2,te.thread_axis("threadIdx.x"))
        if transposeB:
            yy1,yy2=s[BB].split(BB.op.axis[0],factor=32)
            s[BB].bind(yy1, te.thread_axis("blockIdx.y"))
            s[BB].bind(yy2, te.thread_axis("threadIdx.y"))

        x1,x2=s[C].split(C.op.axis[0],factor =32)
        y1,y2=s[C].split(C.op.axis[1],factor=32)
        # s[C].reorder(x1,y1,x2,y2)
        s[C].bind(x1,te.thread_axis("blockIdx.x"))
        s[C].bind(y1,te.thread_axis("blockIdx.y"))
        s[C].bind(x2,te.thread_axis("threadIdx.x"))
        s[C].bind(y2,te.thread_axis("threadIdx.y"))


    # bn = 32
    # CC = s.cache_write(C, 'global')
    # xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    # s[CC].compute_at(s[C], yo)
    # xc, yc = s[CC].op.axis
    # k, = s[CC].op.reduce_axis
    # ko, ki = s[CC].split(k, factor=4)
    # s[CC].reorder(ko, xc, ki, yc)
    # s[CC].unroll(ki)
    # s[CC].vectorize(yc)
    # s[C].parallel(xo)


    # print(tvm.lower(s,[A,B,C],simple_mode=True))

    f=tvm.build(s,[A,B,C],tgt,tgt_host,name=func_name)
    return f



def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""

    #out_size=(input_size-filer_size+2pad)/stride+1
    outShape=(N,M,H-R+1,W-S+1)

    A=te.placeholder(shapeX,dtype=dtype,name="A")
    c=te.reduce_axis((0,C),name="c")
    r=te.reduce_axis((0,R),name="r")
    s=te.reduce_axis((0,S),name="s")
    B=te.placeholder(shapeF,dtype=dtype,name="B")

    C=te.compute(outShape,lambda i,j,h,w:te.sum(A[i,c,h+r,w+s]*B[j,c,r,s],axis=[c,r,s]))

    s=te.create_schedule(C.op)
    if tgt=="cuda":
        xo,yo,xi,yi=s[C].tile(C.op.axis[2],C.op.axis[3],32,32)
        s[C].reorder(xo,yo,xi,yi)
        s[C].bind(xo,te.thread_axis("blockIdx.x"))
        s[C].bind(yo,te.thread_axis("blockIdx.y"))
        s[C].bind(xi,te.thread_axis("threadIdx.x"))
        s[C].bind(yi,te.thread_axis("threadIdx.y"))

    # print (tvm.lower(s,[A,B,C],simple_mode=True))

    f=tvm.build(s,[A,B,C],tgt,tgt_host,name=func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A=te.placeholder(shape,dtype = dtype, name="A")
    '''
    #desined by myself
    k=te.reduce_axis((0,A.shape[1]),name="k")
    A_max=te.compute((A.shape[0],),lambda i:te.max(A[i,k],axis=k))
    A_ex=te.compute(shape,lambda i,j:te.exp(A[i,j]-A_max[i]))
    k1=te.reduce_axis((0,A.shape[1]),name="k1")
    A_ex_sum=te.compute((A.shape[0],),lambda i:te.sum(A_ex[i,k1],axis = k1))
    B=te.compute(shape,lambda i,j:A_ex[i,j]/A_ex_sum[i])

    s=te.create_schedule(B.op)

    if tgt=="cuda":
        s[B].bind(B.op.axis[1],te.thread_axis("threadIdx.x"))
        s[A_ex_sum].bind(k1,te.thread_axis("threadIdx.x"))
        s[A_ex].bind(A_ex.op.axis[1],te.thread_axis("threadIdx.x"))
        s[A_max].bind(k,te.thread_axis("threadIdx.x"))
        # print (tvm.lower(s,[A,B],simple_mode=True))
    '''

    #use topi
    B=topi.nn.softmax(A,axis=1)
    if tgt=="llvm":
        s = te.create_schedule(B.op)
    elif tgt=="cuda":
        # s=topi.cuda.schedule_softmax(B)
        s=te.create_schedule(B.op)
        softmax = B
        expsum = softmax.op.input_tensors[1]
        exp = softmax.op.input_tensors[0]
        max_elem = s[exp].op.input_tensors[1]

        num_thread = 64
        block_x = te.thread_axis("blockIdx.x")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

        s[exp].bind(exp.op.axis[0], block_x)
        s[max_elem].bind(max_elem.op.axis[0], block_x)

        k = expsum.op.reduce_axis[0]
        ko, ki = s[expsum].split(k, factor=num_thread)
        EF = s.rfactor(expsum, ki)
        s[expsum].bind(s[expsum].op.axis[0], block_x)
        s[expsum].bind(s[expsum].op.reduce_axis[0], thread_x)
        s[EF].compute_at(s[expsum], s[expsum].op.reduce_axis[0])
        s[expsum].set_store_predicate(thread_x.var.equal(0))

        tx, xi = s[softmax].split(softmax.op.axis[1], nparts=num_thread)
        s[softmax].bind(softmax.op.axis[0], block_x)
        s[softmax].bind(tx, thread_x)


        print(tvm.lower(s, [A, B], simple_mode=True))
    else:
        s=None

    f=tvm.build(s,[A,B],tgt,tgt_host,name=func_name)
    return f

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    A_=te.placeholder(shape,dtype=dtype,name="A_")
    A=te.placeholder(shape,dtype=dtype,name="A")

    #desined by myself
    k = te.reduce_axis((0, A.shape[1]), name="k")
    A_max = te.compute((A.shape[0],), lambda i: te.max(A[i, k], axis=k))
    A_ex = te.compute(shape, lambda i, j: te.exp(A[i, j] - A_max[i]))
    k1 = te.reduce_axis((0, A.shape[1]), name="k1")
    A_ex_sum = te.compute((A.shape[0],), lambda i: te.sum(A_ex[i, k1], axis=k1))
    A_logsoftmax = te.compute(shape, lambda i, j: te.log(A_ex[i, j] / A_ex_sum[i]))

    k2=te.reduce_axis((0,shape[1]),name="k2")
    A_logsoftmax_sum=te.compute((shape[0],0),lambda i:te.sum(A_logsoftmax[i,k2]*A_[i,k2],axis=k2))
    k3=te.reduce_axis((0,shape[0]),name="k3")
    B=te.compute((1,),lambda i: te.sum(-A_logsoftmax_sum[k3],axis = k3))
    B1=te.compute((1,), lambda i: B[i] / shape[0])

    s=te.create_schedule(B1.op)
    if tgt=="cuda":
        #I'dont know why it can't work?
        s = te.create_schedule(B1.op)

        num_thread = 64
        block_x = te.thread_axis("blockIdx.x")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

        s[A_ex].bind(A_ex.op.axis[0], block_x)
        s[A_max].bind(A_max.op.axis[0], block_x)

        k_ex_sum = A_ex_sum.op.reduce_axis[0]
        ko, ki = s[A_ex_sum].split(k_ex_sum, factor=num_thread)
        EF = s.rfactor(A_ex_sum, ki)
        s[A_ex_sum].bind(s[A_ex_sum].op.axis[0], block_x)
        s[A_ex_sum].bind(s[A_ex_sum].op.reduce_axis[0], thread_x)
        s[EF].compute_at(s[A_ex_sum], s[A_ex_sum].op.reduce_axis[0])
        s[A_ex_sum].set_store_predicate(thread_x.var.equal(0))

        tx, xi = s[A_logsoftmax].split(A_logsoftmax.op.axis[1], nparts=num_thread)
        s[A_logsoftmax].bind(A_logsoftmax.op.axis[0], block_x)
        s[A_logsoftmax].bind(tx, thread_x)

        k_logsoftmax_sum = A_logsoftmax_sum.op.reduce_axis[0]
        klso, klsi = s[A_logsoftmax_sum].split(k_logsoftmax_sum, factor=num_thread)
        lsEF = s.rfactor(A_logsoftmax_sum, klsi)
        s[A_logsoftmax_sum].bind(s[A_logsoftmax_sum].op.axis[0], block_x)
        s[A_logsoftmax_sum].bind(s[A_logsoftmax_sum].op.reduce_axis[0], thread_x)
        s[lsEF].compute_at(s[A_logsoftmax_sum], s[A_logsoftmax_sum].op.reduce_axis[0])
        s[A_logsoftmax_sum].set_store_predicate(thread_x.var.equal(0))

        k_B=B.op.reduce_axis[0]
        kbo,kbi=s[B].split(k_B,factor=num_thread)
        bEF=s.rfactor(B,kbi)
        s[B].bind(s[B].op.reduce_axis[0],thread_x)
        s[bEF].compute_at(s[B],s[B].op.reduce_axis[0])
        s[B].set_store_predicate(block_x.var.equal(0))

        s[B1].set_store_predicate(block_x.var.equal(0))


        print(tvm.lower(s, [A, A_,B1], simple_mode=True))


    f=tvm.build(s,[A,A_,B1],tgt,tgt_host,name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = te.create_schedule(C.op)

    if tgt=="cuda":
        bx,tx=s[C].split(C.op.axis[1],factor=32)
        s[C].bind(bx,te.thread_axis("blockIdx.x"))
        s[C].bind(tx,te.thread_axis("threadIdx.x"))
        # print(tvm.lower(s, [A, C], simple_mode=True))


    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = te.create_schedule(C.op)
    if tgt=="cuda":
        bx,tx=s[C].split(C.op.axis[1],factor=32)
        s[C].bind(bx,te.thread_axis("blockIdx.x"))
        s[C].bind(tx,te.thread_axis("threadIdx.x"))
        # print(tvm.lower(s, [A, C], simple_mode=True))

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = te.placeholder(shape, dtype=dtype, name="A")
    grad = te.placeholder(shape, dtype=dtype, name="grad")
    Y = te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = te.create_schedule(Y.op)
    if tgt=="cuda":
        bx,tx=s[Y].split(Y.op.axis[1],factor=32)
        s[Y].bind(bx,te.thread_axis("blockIdx.x"))
        s[Y].bind(tx,te.thread_axis("threadIdx.x"))
        # print(tvm.lower(s, [X, grad,Y], simple_mode=True))

    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f