defmodule DL do
  #import Nx.Defn
 # Nx.default_backend(Torchx)
 def error_grad(target,input) do
  Matrex.subtract(target,input)
 end
 def error_nn(target,input) do
   Matrex.sum(Matrex.square(Matrex.subtract(target,input)))
 end
 def relu(x) do
    Matrex.apply(x,fn(n)-> if (n>0) do n else 0 end end)
  end
  def relu2deriv(output) do
    Matrex.apply(output,fn(n)-> if (n>0) do 1 else 0 end end)
    #Nx.greater(output,0)
  end
  def sigmoid(x) do
     Matrex.apply(x,:sigmoid)
  end
  def sig2deriv(output) do
    Matrex.multiply(output,Matrex.subtract(1,output))
  end
  def tanh(input) do
    Matrex.apply(input,:tanh)
  end
  def tanh2deriv(output) do
    Matrex.subtract(1,Matrex.square(output))
  end



  def genNewWeights(weights,lr,layer,der) do
    Matrex.add(weights,Matrex.multiply(lr,Matrex.dot(Matrex.transpose(layer),der)))
  end
  def fitWeights(w) do
    r=Matrex.subtract(Matrex.multiply(0.02,w),0.01)
    #r=Matrex.subtract(Matrex.multiply(2,w),1)
    #IO.inspect r
    r
  end
  def newDenseLayer(x,y,_type) do
    fitWeights(Matrex.random(x, y))
   end
  def check_correct(1,output,target) do
    if (Matrex.argmax(Matrex.row(output,1))==Matrex.argmax(Matrex.row(target,1))) do 1 else 0 end
  end
  def check_correct(n,output,target) do
    #IO.puts n
    r1= if (Matrex.argmax(Matrex.row(output,n))==Matrex.argmax(Matrex.row(target,n))) do 1 else 0 end
    r2 = check_correct(n-1,output,target)
    r1+r2
  end
  def predict(input,[_w]) do
    input #Matrex.dot(input,w)
  end
  def predict(input,[{:dropout,_p}|tl]) do
    predict(input,tl)
  end
  def predict(input,[{activation,_derivative}|tl]) do
    o =  activation.(input)
    predict(o,tl)
  end
  def predict(input,[w|t]) do
    o =  Matrex.dot(input,w)
    predict(o,t)
  end
  def test(1,input,target,nn) do
    i1 = Matrex.row(input,1)
    t1 = Matrex.row(target,1)
    o = predict(i1,nn)
    if (Matrex.argmax(t1)==Matrex.argmax(o)) do 1 else 0 end
  end
  def test(n,input,target,nn) do
    i1 = Matrex.row(input,n)
    t1 = Matrex.row(target,n)
    o = predict(i1,nn)
    c1 = if (Matrex.argmax(t1)==Matrex.argmax(o)) do 1 else 0 end
    c2 = test(n-1,input,target,nn)
    c1 + c2
  end
  def run_test(size,input,target,nn)do
    correct = test(size,input,target,nn)
    IO.puts ("Test accuracy: #{ correct/size}")
  end
  def run_net_batch(batchsize,input,[{egrad,enn}],target,_lr) do
    #IO.inspect(weights_l)
    #raise "hell"
    #o = dotPSize(5,input,weights_l)
    #o = Matrex.dot(input,weights_l)

    diff = egrad.(target,input)

    #{rdiff,cdiff}=Matrex.size(diff)
    #IO.puts ("row d: #{rdiff}   col: #{cdiff}")
    finalDerivative = Matrex.divide(diff,batchsize)
    #finalDerivative = Matrex.divide(Matrex.multiply(o,diff),batchsize)

    error =  enn.(target,input)


    #{l,_c} = Matrex.size(target)

    correct = check_correct(batchsize,input,target)
    #newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    #IO.puts "Weights final 2: #{Matrex.sum(newWeights)}"
    #nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[{egrad,enn}],finalDerivative,error,correct}
  end
  def run_net_batch(batchsize,input,[{:dropout,p}|tl],target,lr) do
    {l,c} = Matrex.size(input)
    dp = Matrex.random(l,c)
    dp = Matrex.apply(dp,fn(n)-> if (n<=p) do 1 else 0 end end)
    dp = Matrex.multiply((1/p),dp)
    o = Matrex.multiply(input, dp)
    {net,deriv,error,correct} = run_net_batch(batchsize,o,tl,target,lr)
    nderiv = Matrex.multiply(dp,deriv)
    {[{:dropout,p}|net],nderiv,error,correct}
  end
  def run_net_batch(batchsize,input,[{activation,derivative}|tl],target,lr) do
    o =  activation.(input)
    {net,deriv,error,correct} = run_net_batch(batchsize,o,tl,target,lr)
    nextDeriv = Matrex.multiply(derivative.(input),deriv)
    {[{activation,derivative}|net],nextDeriv,error,correct}
  end
  def run_net_batch(batchsize,input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o =  Matrex.dot(input,w)
    #o =  NN.sigmoid(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = run_net_batch(batchsize,o,tl,target,lr)
    #myDeriv = Matrex.multiply(wD,relu2deriv(o))
    #myDeriv = Matrex.multiply(wD,sig2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,wD)
    #IO.puts "Weights final 1: #{Matrex.sum(newWeights)}"
    #IO.inspect(Nx.sum(newWeights))
    nextLayerD = Matrex.dot(wD,Matrex.transpose(w))
    {[newWeights|net],nextLayerD,error,correct}
  end
  def loop_batch(1,ntrain,bsize,input,nn,target,lr) do
    {newnet,error,correct}=DL.trainNN_batch(ntrain, bsize,input,nn,target,lr,0,0)
    IO.puts("I #{1} error: #{error/(ntrain*bsize)} Acc: #{correct/(ntrain*bsize)}")
    {newnet,error,correct}
  end
  def loop_batch(n,ntrain, bsize, input,nn,target,lr) do
    {newnet,error,correct}=DL.trainNN_batch(ntrain, bsize,input,nn,target,lr,0,0)
    #raise "hell"
    IO.puts("I #{n} error: #{error/(ntrain*bsize)} Acc: #{correct/(ntrain*bsize)}")
    r = loop_batch(n-1,ntrain,bsize,input,newnet,target,lr)
    r
  end
  def trainNN_batch(1,bsize,input,weights,target,lr,argerror,argacc) do
    {_il,ic}=Matrex.size(input)
    {_tl,tc}=Matrex.size(target)
    inputb = Matrex.submatrix(input,1..bsize,1..ic)
    targetb = Matrex.submatrix(target,1..bsize,1..tc)
    {newNet,_wd,error,acc} = DL.run_net_batch(bsize,inputb,weights,targetb,lr)
    #IO.puts "error: #{error} accuracy: #{acc}"
    {newNet,error+argerror,acc+argacc}
  end
  def trainNN_batch(n,bsize,input,weights,target,lr,argerror,argacc) do
    {_il,ic}=Matrex.size(input)
    {_tl,tc}=Matrex.size(target)
    inputb = Matrex.submatrix(input,((n-1)*bsize)+1..(bsize*n),1..ic)
    targetb = Matrex.submatrix(target,((n-1)*bsize)+1..(bsize*n),1..tc)
    {newNet,_wd,error,acc} = DL.run_net_batch(bsize,inputb,weights,targetb,lr)
    trainNN_batch(n-1,bsize,input,newNet,target,lr,argerror+error, argacc+acc)
  end
  def run_net_batch_par(batchsize,input,[{egrad,enn}],target,_lr) do

    diff = egrad.(target,input)
    finalDerivative = Matrex.divide(diff,batchsize)

    #finalDerivative = Matrex.divide(Matrex.multiply(o,diff),batchsize)

    error =  enn.(target,input)


    {l,_c} = Matrex.size(target)

    correct = check_correct(l,input,target)

    #newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    #IO.puts "Weights final 2: #{Matrex.sum(newWeights)}"
    #nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[{egrad,enn}],finalDerivative,error,correct}
  end
  def run_net_batch_par(batchsize,input,[{:dropout,p}|tl],target,lr) do
    #IO.puts "ok"
    {l,c} = Matrex.size(input)
    dp = Matrex.random(l,c)
    dp = Matrex.apply(dp,fn(n)-> if (n<=p) do 1 else 0 end end)
    dp = Matrex.multiply((1/p),dp)
    o = Matrex.multiply(input, dp)
    {net,deriv,error,correct} = run_net_batch_par(batchsize,o,tl,target,lr)
    nderiv = Matrex.multiply(dp,deriv)
    {[{:dropout,p}|net],nderiv,error,correct}
  end
  def run_net_batch_par(batchsize,input,[{activation,derivative}|tl],target,lr) do
    o =  activation.(input)
    {net,deriv,error,correct} = run_net_batch_par(batchsize,o,tl,target,lr)
    nextDeriv = Matrex.multiply(derivative.(input),deriv)
    {[{activation,derivative}|net],nextDeriv,error,correct}
  end
  def run_net_batch_par(batchsize,input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o = Matrex.dot(input,w)
    #o =  NN.sigmoid(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = run_net_batch_par(batchsize,o,tl,target,lr)
    #myDeriv = Matrex.multiply(wD,relu2deriv(o))
    #myDeriv = Matrex.multiply(wD,sig2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = gen_new_weights_par(lr,input,wD)
    #IO.puts "Weights final 1: #{Matrex.sum(newWeights)}"
    #IO.inspect(Nx.sum(newWeights))
    nextLayerD = Matrex.dot(wD,Matrex.transpose(w))
    {[newWeights|net],nextLayerD,error,correct}
  end
  def gen_new_weights_par(lr,layer,der) do
    Matrex.multiply(lr,Matrex.dot(Matrex.transpose(layer),der))
  end
  def input(size) do
    [size]
  end
  def dense(model,size) do
    lsize = List.last(model)
    nm = remove_last(model)
    nm ++ [newDenseLayer(lsize,size,:whatever), size]
  end
  def relu_layer(model) do
    lsize = List.last(model)
    nm = remove_last(model)
    nm ++ [{&DL.relu/1 ,&DL.relu2deriv/1}]++[lsize]
  end
  def sigmoid_layer(model) do
    lsize = List.last(model)
    nm = remove_last(model)
    nm ++ [{&DL.sigmoid/1 ,&DL.sig2deriv/1}]++[lsize]
  end
  def tanh_layer(model) do
    lsize = List.last(model)
    nm = remove_last(model)
    nm ++ [{&DL.tanh/1 ,&DL.tanh2deriv/1}]++[lsize]
  end
  def error(model) do
    nm = remove_last(model)
    nm ++ [{&DL.error_grad/2,&DL.error_nn/2}]
   end
  def dropout(model,p) do
    lsize = List.last(model)
    nm = remove_last(model)
    nm ++ [{:dropout,p}] ++ [lsize]
  end
  def remove_last([_a]) do
    []
  end
  def remove_last([h|t]) do
  [h|remove_last(t)]
 end

end
