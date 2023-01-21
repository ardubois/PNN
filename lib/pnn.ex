defmodule PNN do
  #import Nx.Defn
 # Nx.default_backend(Torchx)
 import NN
 def loopBatch(1,nb,ntrain,slice,input,nn,target,lr) do
  {newnet,wd,error,correct}=trainPBatch(nb,ntrain,slice,input,nn,target,lr,0,0)
  IO.puts("I #{1} error: #{error/(nb*ntrain*slice)} Acc: #{correct/(nb*ntrain*slice)}")
  {newnet,error,correct}
end
def loopBatch(n,nb,ntrain, slice,input,nn,target,lr) do
  {newnet,wd,error,correct}=trainPBatch(nb,ntrain,slice,input,nn,target,lr,0,0)
 # IO.puts "Error"
  #IO.inspect error
  #raise "hell"
  IO.puts("I #{n} error: #{(error)/(nb*ntrain*slice)} Acc: #{correct/(nb*ntrain*slice)}")
  r = loopBatch(n-1,nb,ntrain,slice,input,newnet,target,lr)
  r
end
#def trainPBatch(1,ntrain,slice,input,nn,target,lr) do
#    {newnet,wd1,error,correct}= run_batchWL(ntrain,slice,input,nn,target,lr)
#    #IO.puts("I #{1} error: #{error/(ntrain*slice)} Acc: #{correct/(ntrain*slice)}")
#    {newnet,wd1,error,correct}
#  end
def trainPBatch(1,bsize,slice,input,nn,target,lr, argerror, argacc) do
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    inputb = Matrex.submatrix(input,1..(bsize*slice),1..ic)
    targetb = Matrex.submatrix(target,1..(bsize*slice),1..tc)
    {newnet,wd1,error,correct}= run_batchWL(bsize,slice,inputb,nn,targetb,lr)
    {newnet,wd1,error+argerror,correct+argacc}
  end
def trainPBatch(n,bsize, slice,input,nn,target,lr,argerror, argacc) do
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    inputb = Matrex.submatrix(input,((n-1)*bsize*slice)+1..(slice*bsize*n),1..ic)
    targetb = Matrex.submatrix(target,((n-1)*bsize*slice)+1..(slice*bsize*n),1..tc)
    {newnet,wd1,error,correct}=run_batchWL(bsize,slice,inputb,nn,targetb,lr)

    trainPBatch(n-1,bsize,slice,input,newnet,target,lr,error+argerror, correct+argacc)

  end
  #def trainPBatch(n,ntrain, slice,input,nn,target,lr) do
  #  {newnet,wd1,error,correct}=run_batchWL(ntrain,slice,input,nn,target,lr)
#
#    {il,ic}=Matrex.size(input)
#    {tl,tc}=Matrex.size(target)

 #   restinput = Matrex.submatrix(input,((ntrain*slice)+1)..il,1..ic)
 #   resttarget = Matrex.submatrix(target,((ntrain*slice)+1)..tl,1..tc)

  #  {finalnet,wd2,nerror,nacc} = trainPBatch(n-1,ntrain,slice,restinput,newnet,resttarget,lr)
  #  finalerror = error + nerror
  #  finalacc = correct + nacc

#    {finalnet,wd2,finalerror,finalacc}
#  end
  def dotP(vet,matrix) do
    #{r_,c} = Nx.shape(matrix)
    {r_,c}=Matrex.size(matrix)
  #  #IO.inspect(vet)
  #  #IO.inspect(matrix)
  #  #raise "ok"
     list1 = parallelDot(c,vet,matrix)
  #  #raise "ok"
     list2 = Enum.map(list1,&Task.await/1)
     listf = Enum.map(list2,fn(n) ->  Matrex.at(n,1,1) end)
     Matrex.new([listf])
  end
  def parallelDot(1,vet,matrix) do
    col = Matrex.column(matrix,1)
    task = Task.async(fn -> Matrex.dot(vet,col) end)
    #raise "ok"
    [task]
  end
  def parallelDot(n,vet,matrix) do
    col = Matrex.column(matrix,1)
    {nl,nc}=Matrex.size(matrix)
    restMatrix = Matrex.submatrix(matrix,1..nl,2..nc)
    task = Task.async(fn -> Matrex.dot(vet,col) end)
    tasks = parallelDot(n-1,vet,restMatrix)
    [task|tasks]
  end
  def dotPSize(size,vet,matrix) do
     list_ = parallelDotSize2(size,vet,matrix)
  #  #raise "ok"
     list__ = List.flatten(list_)
     #IO.inspect(list1)
     #raise "list_: #{length list_} list1: #{length list1}"
     list1 = Enum.map(list__,&Task.await/1)
     list2 = List.flatten(list1)
     #IO.inspect list2
     #raise "list2 size: #{length(list2)}"
     listf = Matrex.concat(list2)#Enum.map(list2,fn(n) ->  Matrex.at(n,1,1) end)
     listf
     #Matrex.new([listf])
  end
  def parallelDotSize2(n,vet,matrix) do
    {nl,nc}=Matrex.size(matrix)
    if (nc>n) do
      subMatrix = Matrex.submatrix(matrix,1..nl,1..n)
      #IO.inspect list
      #raise "hell"
      #if (length(list) != 5) do raise "size" end
      task = Task.async(fn -> Matrex.dot(vet,subMatrix) end)
      restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      tasks = parallelDotSize2(n,vet,restMatrix)
      [ task, tasks]
    else
      #list = getColumns(matrix,nc)
      #raise "fuck"
      task = Task.async(fn -> Matrex.dot(vet,matrix) end)
      #restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      #tasks = parallelDotSize(n,vet,restMatrix)
      [task]
    end
  end
  def parallelDotSize(n,vet,matrix) do
    {nl,nc}=Matrex.size(matrix)
    if (nc>n) do
      list = getColumns(matrix,n)
      #IO.inspect list
      #raise "hell"
      #if (length(list) != 5) do raise "size" end
      task = Task.async(fn -> Enum.map(list, fn(col) -> Matrex.dot(vet,col) end) end)
      restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      tasks = parallelDotSize(n,vet,restMatrix)
      [ task, tasks]
    else
      list = getColumns(matrix,nc)
      #raise "fuck"
      task = Task.async(fn -> Enum.map(list, fn(col) -> Matrex.dot(vet,col) end) end)
      #restMatrix = Matrex.submatrix(matrix,1..nl,(n+1)..nc)
      #tasks = parallelDotSize(n,vet,restMatrix)
      [task]
    end
  end
  def getColumns(matrix,1)do
    col = Matrex.column(matrix,1)
    [col]
  end
  def getColumns(matrix,ncol)do
    col = Matrex.column(matrix,1)
    {nl,nc}=Matrex.size(matrix)
    restMatrix = Matrex.submatrix(matrix,1..nl,2..nc)
    cols =getColumns(restMatrix,ncol-1)
    [col|cols]
  end
  def run_batch(size,input,weights,target,lr) do
    tasks = slice_entries_p(size,input,target,weights,lr)
    #tasks = Enum.map(list, fn({i1,t1}) -> Task.async(fn -> NN.runNet(i1,weights,t1,lr)end) end)
    #results = Enum.map(tasks,&Task.await/1)
    [hr|tr] = tasks
    r1 = Task.await(hr)
    List.foldr(tr, r1, fn(task,{nn2,wd2,erro2,acc2}) -> {nn1,wd1,erro1,acc1} = Task.await task
                                                        #IO.inspect erro2
                                                        #IO.inspect acc2
                                                        {sumNNs(nn1,nn2),wd1,erro1+erro2,acc1+acc2} end)
  end
  #def run_batchWL(size,slice,input,weights,target,lr) do
  #  list = slice_entries(size,slice,input,target)
  #  tasks = Enum.map(list, fn({i1,t1}) -> WL.send_job({size,slice,i1,weights,t1,lr}) end)
  #  #results = Enum.map(tasks,&Task.await/1)
  #  #[hr|tr] = tasks
  #  #{nn_,wd_,erro_,acc_} = WL.get_result(hr)
  #  List.foldr(tasks, {weights,0,0,0}, fn(task,{nn2,wd2,erro2,acc2}) -> {nn1,wd1,erro1,acc1} = WL.get_result task
  #                                                      #IO.inspect erro2
  #                                                      #IO.inspect acc2
  #                                                      #IO.inspect erro1
  #                                                      #IO.inspect acc1
  #                                                      {sumNNs(nn1,nn2),wd1,erro1+erro2,acc1+acc2} end)
  #end
  def sumNNs([{a,b}],[w2]) do
    [{a,b}]
  end
  def sumNNs([w1],[w2]) do
    w3 =Matrex.add(w1,w2)
    [w3]
  end
  def sumNNs([{a,b}|t1],[w2|t2]) do
    rest = sumNNs(t1,t2)
    [{a,b}|rest]

  end
  def sumNNs([w1|t1],[w2|t2]) do
    w3 =Matrex.add(w1,w2)
    rest = sumNNs(t1,t2)
    [w3|rest]

  end
  def divNN([w],n) do
    nw =Matrex.divide(w,n)
    [nw]
  end
  def divNN([w|t],n) do
    nw =Matrex.divide(w,n)
    nt = divNN(t,n)
    [nw|nt]
  end
#  def slice_entries(1,slice,input,target)do
#    {il,ic}=Matrex.size(input)
#    {tl,tc}=Matrex.size(target)
#    inputs = Matrex.submatrix(input,1..slice,1..ic)
#    targets = Matrex.submatrix(target,1..slice,1..tc)
#
#    #task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
#    [{inputs,targets}]
#  end
 def run_batchWL(size,slice,input,weights,target,lr) do
  tasks = slice_entries(size, size, slice,input, weights, target,lr)
  #tasks = Enum.map(list, fn({i1,t1}) -> WL.send_job({size,slice,i1,weights,t1,lr}) end)
  #results = Enum.map(tasks,&Task.await/1)
  #[hr|tr] = tasks
  #{nn_,wd_,erro_,acc_} = WL.get_result(hr)
  List.foldr(tasks, {weights,0,0,0}, fn(task,{nn2,wd2,erro2,acc2}) -> {nn1,wd1,erro1,acc1} = WL.get_result task
                                                      #IO.inspect erro2
                                                      #IO.inspect acc2
                                                      #IO.inspect erro1
                                                      #IO.inspect acc1
                                                      {sumNNs(nn1,nn2),wd1,erro1+erro2,acc1+acc2} end)
 end
  def slice_entries(1,size, slice,input,nn, target, lr)do

    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    inputs = Matrex.submatrix(input,1..slice,1..ic)
    targets = Matrex.submatrix(target,1..slice,1..tc)
    task = WL.send_job({size,slice,inputs,nn,targets,lr})
    #task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
    [task]
  end
  def slice_entries(n,size, slice,input,nn,target,lr)do
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    inputs = Matrex.submatrix(input,((n-1)*slice)+1..(slice*n),1..ic)
    targets = Matrex.submatrix(target,((n-1)*slice)+1..(slice*n),1..tc)
    task = WL.send_job({size,slice,inputs,nn,targets,lr})
    tasks = slice_entries(n-1,size, slice,input,nn,target,lr)
    [task|tasks]
  end
  #def slice_entries(n,slice,input,target)do
  #  {il,ic}=Matrex.size(input)
  #  {tl,tc}=Matrex.size(target)
  #  inputs = Matrex.submatrix(input,1..slice,1..ic)
  #  targets = Matrex.submatrix(target,1..slice,1..tc)
  #  restinput = Matrex.submatrix(input,(slice+1)..il,1..ic)
  #  resttarget = Matrex.submatrix(target,(slice+1)..tl,1..tc)
  #  slices = slice_entries(n-1,slice,restinput,resttarget)
  #  [{inputs,targets}|slices]
  #end
  def slice_entries_p(1,input,target,weights,lr)do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
    [task]
  end
  def slice_entries_p(n,input,target,weights,lr)do
    input1 = Matrex.row(input,1)
    target1 = Matrex.row(target,1)
    task =Task.async(fn -> NN.runNet(input1,weights,target1,lr)end)
    {il,ic}=Matrex.size(input)
    {tl,tc}=Matrex.size(target)
    restinput = Matrex.submatrix(input,2..il,1..ic)
    resttarget = Matrex.submatrix(target,2..tl,1..tc)
    slices = slice_entries_p(n-1,restinput,resttarget,weights,lr)
    [task|slices]
  end
end
