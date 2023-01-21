import NN
import PNN
#inputSize = 3
#hiddenSize = 4
#outputSize = 1
#alpha = 0.005
#weights_0_1 = Nx.tensor ( [[-0.16595599,  0.40763847, -0.99977125],
#                            [-0.39533485, -0.70648822, -0.81532281],
#                            [-0.62747958 ,-0.34188906 ,-0.20646505]]) #NN.newDenseLayer(inputSize,hiddenSize,:relu)
#

#weights_1_2 = Nx.tensor([[ 0.07763347],
#                          [-0.16161097],
#                          [ 0.370439  ]])#NN.newDenseLayer(hiddenSize,outputSize,:relu)


#nn = [weights_0_1,weights_1_2]

#sl_input = Matrex.new([  [ 1, 0, 1],
#                        [ 0, 1, 1],
#                        [ 0, 0, 1],
#                        [ 1, 1, 1] ])

#sl_target = Matrex.transpose(Matrex.new([[1, 1, 0, 0]]))



w0_ = Matrex.load("w01.csv")
w1_ = Matrex.load("w12.csv")
nn = [w0_,w1_]

inputSize = 784 #pixels per image
hiddenSize = 180#40#360#180#40
outputSize = 10
alpha =  0.01
#nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
 #     NN.newDenseLayer(hiddenSize,outputSize,:relu)]

#nn = [NN.newDenseLayer(inputSize,256,:relu),
#      NN.newDenseLayer(256,180,:relu),
#      NN.newDenseLayer(180,40,:relu),
#      NN.newDenseLayer(40,10,:relu)]

#nn= [NN.newDenseLayer(inputSize,100,:relu),
#      NN.newDenseLayer(100,10,:relu)]

      #NN.newDenseLayer(80,10,:relu)]
     # NN.newDenseLayer(500,10,:relu)]

#nn = [NN.newDenseLayer(inputSize,200,:relu),
#      NN.newDenseLayer(200,80,:relu),
#      NN.newDenseLayer(80,10,:relu)]


#nn = [NN.newDenseLayer(inputSize,500,:relu),
 #    NN.newDenseLayer(500,180,:relu),
  #   NN.newDenseLayer(180,40,:relu),
   #  NN.newDenseLayer(40,10,:relu)]

#[wh|wt] =nn


images = Matrex.load("mnistfullIMG.mtx")
labels = Matrex.load("mnistfullLAB.mtx")

#Matrex.save(images, "mnistfullIMG.mtx")
#Matrex.save(labels, "mnistfullLAB.mtx")

#images = Matrex.load("imgMNIST.csv")
#labels = Matrex.load("tarMNIST.csv")

#images = Matrex.load("img5000.csv")
#labels = Matrex.load("lab5000.csv")
IO.puts "finish loading"

#images = Matrex.submatrix(images,1..1000,1..784)
#labels = Matrex.submatrix(labels,1..1000,1..10)
#images = Matrex.load("fashionIMG.csv")
#labels = Matrex.load("fashionLAB.csv")

#IO.puts Matrex.size(images)
#IO.puts Matrex.size(labels)
#raise "hell"

#{newNet,d,errorFinal,correct} = NN.runNet(input1,nn2,target1,alpha)
#time1 = Time.utc_now()
#{newNet,errorFinal,correct} = NN.trainNN(1000,images,nn,labels,alpha)

#time1 = Time.utc_now()
#{time,r} = :timer.tc(&Bench.execNNp/6,[100,[],input1,nn,target1,alpha])
#{time,r} = :timer.tc(&Bench.execNN/5,[100,input1,nn,target1,alpha])
#IO.puts ("time: #{(time)/(1_000_000)}")



#{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[1,1000,images,nn,labels,alpha])

#WL.testSystem()

#{time,r}=:timer.tc(&NN.run_batchWL/5,[100,images,nn,labels,alpha])


#{time,{newnet,error,correct}}=:timer.tc(&NN.trainNN/5,[100,images,nn,labels,alpha])
#IO.puts ("time: #{time/(1_000_000)}")


#nn =
#      DL.input(784)
#      |> DL.dense(500)
#      |> DL.reluLayer()
#      |> DL.dense(180)
#      |> DL.reluLayer()
#      |> DL.dense(10)
#      |> DL.error()


#IO.inspect (nn)
#raise "hell"

nn = [w0_, {&DL.relu/1, &DL.relu2deriv/1}, w1_, {&DL.error_grad/2, &DL.error_nn/2}]

WL.testSystem(10)

#{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[10,100,images,nn,labels,alpha])

{time,{newnet,errorFinal,correct} } = :timer.tc(&PNN.loopBatch/8,[10,600,10,10,images,nn,labels,alpha])


#{time,{newnet,error,acc}} = :timer.tc(&NN.loop_batch/7,[10,600,100,images, nn, labels, alpha])

#{time,{newnet,error,acc}} = :timer.tc(&DL.loop_batch/7,[10,600,100,images, nn, labels, alpha])

#{time,{newNet,wd,errorFinal,correct} } = :timer.tc(&NN.trainPBatch/6,[10,100,images,nn,labels,alpha])

#{time,{newnet,error,correct}}=:timer.tc(&NN.trainNN/5,[1000,images,nn,labels,alpha])

IO.puts ("time: #{time/(1_000_000)}")

timages = Matrex.load("test_imgFULL.mtx")
ttarget = Matrex.load("test_labFULL.mtx")

#Matrex.save(timages, "test_imgFULL.mtx")
#Matrex.save(ttarget, "test_labFULL.mtx")

{time, _k } = :timer.tc(&DL.run_test/4, [10000,timages,ttarget, newnet])

IO.puts ("time: #{time/(1_000_000)}")

#{newnet,wd,error,acc}=NN.run_batch(100,images,nn,labels,alpha)
#[w1,w2]= nn

#{l1,c1}=Matrex.size(w1)
#{l2,c2}=Matrex.size(w2)
#IO.puts "w1 = #{l1} x #{c1}"
#IO.puts "w1 = #{l2} x #{c2}"
#IO.puts ("weights 1: #{Matrex.sum(w1)}")
#IO.puts ("weights 2: #{Matrex.sum(w2)}")
#IO.puts("inicio")
#{newnet,error,acc} = NN.trainNN_batch(10,100,images, nn, labels, alpha)

#{newnet,error,acc} = NN.loop_batch(1000,1000,1,images, nn, labels, alpha)

#{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[100,1000,images,nn,labels,alpha])
#IO.puts ("time: #{time/(1_000_000)}")


#IO.puts error
#IO.puts acc

#timef = Time.diff(time2,time1)

Process.exit(self(),:ok)





#IO.inspect timef
#IO.puts("error")
#IO.inspect errorFinal
#IO.puts("Acc")
#IO.inspect(correct)
