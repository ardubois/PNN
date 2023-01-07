defmodule WL do
import PNN
  def work_list_server(n) do
    receive do
      {:addWork, clientpid, work} ->
        send(clientpid, {:workAdded , n})
        receive do
          {:idle, workerpid} ->
            send(workerpid,{:work, clientpid, n, work})
            work_list_server(n+1)
        end
    end
  end
  def send_job(work) do
    #send({:work_list_server,:"main@Satanas-666"},{:addWork, self(),work})
    send(:work_list_server,{:addWork, self(),work})
    receive do
      {:workAdded , n} -> n
    end
  end
  def get_result(workid) do
     receive do
        {:workresult,workid,r} -> r
     end
  end
  def init_work_list_server() do
    pid = spawn_link(fn -> work_list_server(0) end)
    Process.register(pid, :work_list_server)
  end
  def worker() do
    #send({:work_list_server,:"main@Satanas-666"}, {:idle, self()})
    send(:work_list_server, {:idle, self()})
    receive do
       {:work, clientpid, workid,{size ,input1,weights,target1,lr}} ->
              {net,wd,error,acc}=NN.runNet( input1,weights,target1,lr)
              newNet = PNN.divNN(net,size)
              nwd = wd/size
              nerror = error/size
              nacc = acc/size
              send(clientpid,{:workresult, workid, {newNet,nwd,nerror,nacc}})
              worker()
    end
  end
  def init_workers(1) do
    spawn_link(fn -> WL.worker()end)
  end
  def init_workers(n) do
    spawn_link(fn -> WL.worker()end)
    init_workers(n-1)
  end
  def testSystem(n)do
    WL.init_work_list_server()
    init_workers(n)
   # Node.spawn_link(:"core1@Satanas-666",fn -> WL.worker()end)
   # Node.spawn_link(:"core2@Satanas-666",fn -> WL.worker()end)
   # Node.spawn_link(:"core3@Satanas-666",fn -> WL.worker()end)

  end
  def test_cores()do
    WL.init_work_list_server()
    Node.spawn_link(:"core1@Satanas-666",fn -> WL.worker()end)
    Node.spawn_link(:"core2@Satanas-666",fn -> WL.worker()end)
    Node.spawn_link(:"core3@Satanas-666",fn -> WL.worker()end)
    Node.spawn_link(:"core4@Satanas-666",fn -> WL.worker()end)
    Node.spawn_link(:"core5@Satanas-666",fn -> WL.worker()end)
    Node.spawn_link(:"core6@Satanas-666",fn -> WL.worker()end)
  end
def test() do
    inputSize = 784 #pixels per image
    hiddenSize = 180#360#180#40#720#360#40#360#180#40
    outputSize = 10
    alpha = 0.005
    nn = [NN.newDenseLayer(inputSize,hiddenSize,:relu),
          NN.newDenseLayer(hiddenSize,outputSize,:relu)]

    images = Matrex.load("imgMNIST.csv")
    labels = Matrex.load("tarMNIST.csv")

    input1 = images[1]
    target1 = labels[1]

    WL.testSystem(7)
    #WL.test_cores()
    #{time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loop/6,[100,1000,images,nn,labels,alpha])

    {time,{newNet,errorFinal,correct} } = :timer.tc(&NN.loopBatch/7,[100,10,100,images,nn,labels,alpha])
    #{time,{newNet,wd,errorFinal,correct} } = :timer.tc(&NN.trainPBatch/6,[10,100,images,nn,labels,alpha])

    #{time,{newnet,error,correct}}=:timer.tc(&NN.trainNN/5,[1000,images,nn,labels,alpha])

    IO.puts ("time: #{time/(1_000_000)}")


    #timef = Time.diff(time2,time1)

    Process.exit(self(),:ok)


   end


end

#WL.test()
