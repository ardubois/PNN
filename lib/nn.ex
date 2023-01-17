#Nx.default_backend(Torchx.Backend)
#Nx.default_backend(EXLA.Backend)
# Sets the global compilation options
#Nx.Defn.global_default_options(compiler: EXLA)
# Sets the process-level compilation options
#Nx.Defn.default_options(compiler: EXLA)
defmodule NN do
  #import Nx.Defn
 # Nx.default_backend(Torchx)
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
  def runNet(input,[weights_l],target,lr) do
    #IO.inspect(weights_l)
    #raise "hell"
    #o = dotPSize(5,input,weights_l)
    o = Matrex.dot(input,weights_l)
    finalDerivative = Matrex.subtract(target,o)
    #finalDerivative = Matrex.multiply(o,Matrex.subtract(target,o))

    error =  Matrex.sum(Matrex.square(finalDerivative))
    correct = if (Matrex.argmax(o)==Matrex.argmax(target)) do 1 else 0 end
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def runNet(input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o =  relu(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = runNet(o,tl,target,lr)
    myDeriv = Matrex.multiply(wD,relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    #IO.inspect(Nx.sum(newWeights))
    nextLayerD = Matrex.dot(myDeriv,Matrex.transpose(w))
    #IO.puts "Weights final 1: #{Matrex.sum(newWeights)}"
    {[newWeights|net],nextLayerD,error,correct}
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
  def newDenseLayer(x,y,type) do
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
  def predict(input,[w]) do
    Matrex.dot(input,w)
  end
  def predict(input,[w|t]) do
    o =  relu(Matrex.dot(input,w))
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
  def run_net_batch(batchsize,input,[weights_l],target,lr) do
    #IO.inspect(weights_l)
    #raise "hell"
    #o = dotPSize(5,input,weights_l)
    o = Matrex.dot(input,weights_l)
    diff = Matrex.subtract(target,o)
    #{rdiff,cdiff}=Matrex.size(diff)
    #IO.puts ("row d: #{rdiff}   col: #{cdiff}")
    finalDerivative = Matrex.divide(diff,batchsize)
    #finalDerivative = Matrex.divide(Matrex.multiply(o,diff),batchsize)

    error =  Matrex.sum(Matrex.square(diff))


    #{l,_c} = Matrex.size(target)

    correct = check_correct(batchsize,o,target)
    newWeights = genNewWeights(weights_l,lr,input,finalDerivative)
    #IO.puts "Weights final 2: #{Matrex.sum(newWeights)}"
    nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def run_net_batch(batchsize,input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o =  relu(Matrex.dot(input,w))
    #o =  NN.sigmoid(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = run_net_batch(batchsize,o,tl,target,lr)
    myDeriv = Matrex.multiply(wD,relu2deriv(o))
    #myDeriv = Matrex.multiply(wD,sig2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = genNewWeights(w,lr,input,myDeriv)
    #IO.puts "Weights final 1: #{Matrex.sum(newWeights)}"
    #IO.inspect(Nx.sum(newWeights))
    nextLayerD = Matrex.dot(myDeriv,Matrex.transpose(w))
    {[newWeights|net],nextLayerD,error,correct}
  end
  def run_net_batch_par(batchsize,input,[weights_l],target,lr) do
    #IO.inspect(weights_l)
    #raise "hell"
    #o = dotPSize(5,input,weights_l)
    o = Matrex.dot(input,weights_l)
    diff = Matrex.subtract(target,o)
    #{rdiff,cdiff}=Matrex.size(diff)
    #IO.puts ("row d: #{rdiff}   col: #{cdiff}")
    finalDerivative = Matrex.divide(diff,batchsize)
    #finalDerivative = Matrex.divide(Matrex.multiply(o,diff),batchsize)

    error =  Matrex.sum(Matrex.square(diff))


    {l,_c} = Matrex.size(target)

    correct = check_correct(l,o,target)
    newWeights = gen_new_weights_par(lr,input,finalDerivative)
    #IO.puts "Weights final 2: #{Matrex.sum(newWeights)}"
    nextLayerD = Matrex.dot(finalDerivative,Matrex.transpose(weights_l))
    {[newWeights],nextLayerD,error,correct}
  end
  def run_net_batch_par(batchsize,input,[w|tl],target,lr) do
    #o =  relu(Matrex.dot(input,w))
    #IO.inspect(w)
    #raise "hell"
    o =  relu(Matrex.dot(input,w))
    #o =  NN.sigmoid(Matrex.dot(input,w))
    #IO.inspect o
    #raise "oi"
    {net,wD,error,correct} = run_net_batch_par(batchsize,o,tl,target,lr)
    myDeriv = Matrex.multiply(wD,relu2deriv(o))
    #myDeriv = Matrex.multiply(wD,sig2deriv(o))
    #IO.inspect(relu2deriv(o))
    newWeights = gen_new_weights_par(lr,input,myDeriv)
    #IO.puts "Weights final 1: #{Matrex.sum(newWeights)}"
    #IO.inspect(Nx.sum(newWeights))
    nextLayerD = Matrex.dot(myDeriv,Matrex.transpose(w))
    {[newWeights|net],nextLayerD,error,correct}
  end
  def gen_new_weights_par(lr,layer,der) do
    Matrex.multiply(lr,Matrex.dot(Matrex.transpose(layer),der))
  end
  def trainNN(1, input, nn,target,lr) do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    {net,wD,error,correct} = runNet(input1,nn,target1,lr)
    {net,error,correct}
  end
  def trainNN(n,input,nn,target,lr) do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    {net,wD,newError,correct} = runNet(input1,nn,target1,lr)

    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    if (il == 2) do
        tinput  = Matrex.row(input,2) # Drop the first "row"
        ttarget = Matrex.row(target,2)
        {finalNet,errorsofar,correctsofar} = trainNN(n-1,tinput,net,ttarget,lr)
        myError = newError+errorsofar
        myCorrect = correct+correctsofar# correct+correctsofar
        {finalNet,myError,myCorrect}
    else
        tinput  = input[2..il] # Drop the first "row"
        ttarget = target[2..tl] # Drop the first "row"
        {finalNet,errorsofar,correctsofar} = trainNN(n-1,tinput,net,ttarget,lr)
        myError = newError+errorsofar
        myCorrect = correct+correctsofar# correct+correctsofar
        {finalNet,myError,myCorrect}
    end
  end
  def loop(1,ntrain,input,nn,target,lr) do
    {newnet,error,correct}=trainNN(ntrain,input,nn,target,lr)
    IO.puts("I #{1} error: #{error/ntrain} Acc: #{correct/ntrain}")
    {newnet,error,correct}
  end
  def loop(n,ntrain, input,nn,target,lr) do
    {newnet,error,correct}=trainNN(ntrain,input,nn,target,lr)
   # IO.puts "Error"
    #IO.inspect error
    IO.puts("I #{n} error: #{(error)/ntrain} Acc: #{correct/ntrain}")
    r = loop(n-1,ntrain,input,newnet,target,lr)
    r
  end
  def loop_batch(1,ntrain,bsize,input,nn,target,lr) do
    {newnet,error,correct}=trainNN_batch(ntrain, bsize,input,nn,target,lr)
    IO.puts("I #{1} error: #{error/(ntrain*bsize)} Acc: #{correct/(ntrain*bsize)}")
    {newnet,error,correct}
  end
  def loop_batch(n,ntrain, bsize, input,nn,target,lr) do
    {newnet,error,correct}=trainNN_batch(ntrain, bsize,input,nn,target,lr)
    #raise "hell"
    IO.puts("I #{n} error: #{error/(ntrain*bsize)} Acc: #{correct/(ntrain*bsize)}")
    r = loop_batch(n-1,ntrain,bsize,input,newnet,target,lr)
    r
  end
  def trainNN_batch(1,bsize,input,weights,target,lr) do
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    #IO.puts "lac #{1}"
    inputb = Matrex.submatrix(input,1..bsize,1..ic)
    targetb = Matrex.submatrix(target,1..bsize,1..tc)
    {newNet,wd,error,acc} = NN.run_net_batch(bsize,inputb,weights,targetb,lr)
    #IO.puts "error: #{error} accuracy: #{acc}"
    {newNet,error,acc}
  end
  def trainNN_batch(n,bsize,input,weights,target,lr) do
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    #IO.puts "lac #{n}"
    inputb = Matrex.submatrix(input,1..bsize,1..ic)
    targetb = Matrex.submatrix(target,1..bsize,1..tc)
    {newNet,wd,error,acc} = NN.run_net_batch(bsize,inputb,weights,targetb,lr)
    #IO.puts "error: #{error} accuracy: #{acc}"
    #[w1,w2] = newNet
    #IO.puts "w1: #{Matrex.sum(w1)}    w2: #{Matrex.sum(w2)}"
    #raise "hell"
    inputr = Matrex.submatrix(input,(bsize+1)..il,1..ic)
    targetr = Matrex.submatrix(target,(bsize+1)..tl,1..tc)
    {finalnet,nerror,nacc}= trainNN_batch(n-1,bsize,inputr,newNet,targetr,lr)
    finalerror = error + nerror
    finalacc = acc + nacc
    #IO.puts "error: #{finalerror} accuracy: #{finalacc}"
    {finalnet,finalerror,finalacc}
  end
end

defmodule Bench do
  def execNN(1,input1,nn,target1,alpha) do
    #task1 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    #task2 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    #task3 = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    NN.runNet(input1,nn,target1,alpha)
    #Task.await(task1)
    #Task.await(task2)
    #Task.await(task3)
  end
  def execNN(n,input1,nn,target1,alpha) do
    NN.runNet(input1,nn,target1,alpha)
    execNN(n-1,input1,nn,target1,alpha)
  end
  def execNNp(1,l,input1,nn,target1,alpha) do
    task = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    Enum.map([task|l],&Task.await/1)
    #Task.await(task1)
    #Task.await(task2)
    #Task.await(task3)
  end
  def execNNp(n,l,input1,nn,target1,alpha) do
    task = Task.async(fn -> NN.runNet(input1,nn,target1,alpha) end)
    execNNp(n-1,[task|l],input1,nn,target1,alpha)
  end
end
