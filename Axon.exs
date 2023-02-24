Mix.install([
  {:req, "~> 0.3"},
  {:axon,"~>0.3.0"}, # github: "elixir-nx/axon"},
  {:nx, "~> 0.4.0"},
  {:torchx, "~> 0.4.0"},
  {:exla, "~> 0.4"}
])

# Global default backend
#Nx.global_default_backend(EXLA.Backend)
# Process default backend
#Nx.default_backend(Torchx.Backend)
# Sets the global compilation options
#Nx.Defn.global_default_options(compiler: EXLA)
# Sets the process-level compilation options
#Nx.Defn.default_options(compiler: Torchx)
Nx.Defn.default_options(compiler: EXLA)
#base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
#%{body: train_images} = Req.get!(base_url <> "train-images-idx3-ubyte.gz")
#%{body: train_labels} = Req.get!(base_url <> "train-labels-idx1-ubyte.gz")
#t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
#t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

{:ok,file1}= File.open("train-images-idx3-ubyte.gz", [:read, :compressed,:raw])
train_images= IO.binread(file1,:all)
{:ok,file2}= File.open("train-labels-idx1-ubyte.gz", [:read, :compressed,:raw])
train_labels = IO.binread(file2,:all)




<<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> = train_images
<<_::32, n_labels::32, labels::binary>> = train_labels
#targets_ = Nx.to_batched(Nx.from_numpy("labelsMNIST.npy"),1)
#images_ = Nx.to_batched(Nx.from_numpy("imagesMNIST.npy"),1)

#t_ = Nx.from_numpy("labelsMNIST.npy")
#i_ = Nx.from_numpy("imagesMNIST.npy")

images =
  images
  |> Nx.from_binary({:u, 8})
  |> Nx.reshape({n_images, 1, n_rows, n_cols}, names: [:images, :channels, :height, :width])
  |> Nx.divide(255)
  |> Nx.to_batched(512)

targets =
    labels
    |> Nx.from_binary({:u, 8})
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched(512)
#targets = targets[0..4999]

#targets = Nx.as_type(targets,:u64)
#IO.inspect targets


model =
  Axon.input("input", shape: {nil, 1, 28, 28})
  |> Axon.flatten()
  |> Axon.dense(512)
  |> Axon.relu()
  #|> Axon.dropout(rate: 0.5)
  |> Axon.dense(256)
  |> Axon.relu()
  #|> Axon.dense(128, activation: :relu)
 # |> Axon.dropout(rate: 0.5)
  |> Axon.dense(10)


params =
    model
    #|> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
    |> Axon.Loop.trainer(:mean_squared_error, Axon.Optimizers.sgd(0.1))
   # |> Axon.Loop.trainer(Axon.Losses.mean_squared_error/2, Axon.Optimizers.sgd(0.01))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    #|> Axon.Loop.run(Stream.zip(images, targets), %{}, epochs: 10)

params = Axon.Loop.run(params,Stream.zip(images, targets), %{}, epochs: 10)



#{time,params} = :timer.tc(&Axon.Loop.run/4,[params,Stream.zip(images, targets), %{}, epochs: 10])

#IO.puts ("time: #{time/(1_000_000)}")

#first_batch = Enum.at(images, 0)

#IO.inspect (first_batch)

#output = Axon.predict(model, params, first_batch)

#IO.inspect output

#IO.inspect( Nx.argmax(output, axis: 1))
