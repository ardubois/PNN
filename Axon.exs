Mix.install([
  {:req, "~> 0.3"},
  {:axon, github: "elixir-nx/axon"},
  {:nx, "~> 0.4.0"},
  {:torchx, "~> 0.4.0"},
  {:exla, "~> 0.4"}
])

# Global default backend
#Nx.global_default_backend(EXLA.Backend)
# Process default backend
Nx.default_backend(Torchx.Backend)
# Sets the global compilation options
#Nx.Defn.global_default_options(compiler: EXLA)
# Sets the process-level compilation options
#Nx.Defn.default_options(compiler: Torchx)
Nx.Defn.default_options(compiler: EXLA)
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
%{body: train_images} = Req.get!(base_url <> "train-images-idx3-ubyte.gz")
%{body: train_labels} = Req.get!(base_url <> "train-labels-idx1-ubyte.gz")

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


# = images[0..4999]
images = Nx.as_type(images,:u64)
images =
  images
  |> Nx.to_batched(100)
IO.inspect images
targets =
    labels
    |> Nx.from_binary({:u, 8})
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

#targets = targets[0..4999]

targets = Nx.as_type(targets,:u64)
IO.inspect targets
targets =
  targets
    |> Nx.to_batched(100)

IO.inspect images
IO.inspect targets
IO.puts("images")
#IO.inspect(i_)
IO.inspect(images)
model =
  #Axon.input("input", shape: {nil,784})
  Axon.input("input", shape: {nil, 1, 28, 28})
  |> Axon.flatten()
  |> Axon.dense(256, activation: :relu)
  #|> Axon.dense(80, activation: :sigmoid)
  |> Axon.dense(40, activation: :relu)
  |> Axon.dense(10, activation: :softmax)

time1 = Time.utc_now()
params =
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(images, targets), %{}, epochs: 100)

time2 = Time.utc_now()

timef = Time.diff(time2,time1)

IO.puts timef
