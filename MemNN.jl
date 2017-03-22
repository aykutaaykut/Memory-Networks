using Knet, AutoGrad, Compat, ArgParse

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    ("--datafiles"; nargs = '+'; help = "If provided, use first file for training, second for dev, others for test.")
    ("--input"; arg_type = Char; default = 's'; help = "s for sentence sequences, w for word sequences as input")
    ("--batchsize"; arg_type=Int; default=90; help="Number of sequences to train on in parallel.")
    ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
    ("--seed"; arg_type=Int; default=38; help="Random number seed.")
    ("--atype"; default = (gpu() >= 0 ? "KnetArray{Float32}" : "Array{Float32}"); help = "Array type: Array for CPU, KnetArray for GPU")
  end
  return parse_args(s; as_symbols = true)
end

function main(args = ARGS)
  #Fixed configurations for baseline model
  embedding_dimension = 100
  learning_rate = 0.01
  margin = 0.1
  total_epochs = 10

  #User settings are added.
  settings = parse_commandline()
  println("settings", [(symbol, value) for (symbol, value) in settings]...)
  settings[:seed] > 0 && srand(opts[:seed])
  settings[:atype] = eval(parse(settings[:atype]))

  #Classification of datafiles as training, dev and test.
  training_data = settings[:datafiles][1]
  dev_data = settings[:datafiles][2]
  test_data = settings[:datafiles][3:end]

  #Memory of the model.
  memory = Any[]

  #Creating the dictionary of words in the data.
  dict = createDict(training_data)
  dict_length = length(dict)
  info("$dict_length unique words.")
  feature_space = 3 * dict_length

  model = initWeights(settings[:atype], feature_space, embedding_dimension, settings[:winit])

end

function createDict(training_data)
  dict = Dict{String, Any}
  number = 1
  open(training_data) do f
    while !eof(f)
      str = readline(f)
      number, dict = parseLineAddDict(number, str, dict)
    end
  end
  return dict
end

function parseLineAddDict(number, line, dict)
  words = split(line)
  for str in words
    if !haskey(dict, str)
      dict[str] = [number, (number + 1), (number + 2)]
      number = number + 3
    end
  end
  return number, dict
end

function initWeights(atype, feature_space, embedding_dimension, winit)
  weights = Dict()

  uo = winit * randn(embedding_dimension, feature_space)
  ur = winit * randn(embedding_dimension, feature_space)
  weights[:uo] = uo
  weights[:ur] = ur

  for k in keys(weights)
    weights[k] = convert(atype, weights[k])
  end
  return weights
end

function I(x)
  return x
end

function G(x, memory)
  feature_rep = I(x)
  push!(memory, feature_rep)
  return memory
end

function phi(x, d)

end

function s(x, y, u)
  score = 0
  phiy = phi(y, d)
  for input in x
    phix = phi(input, d)
    current_score = phix' * u' * u * phiy
    score = score + current_score
  end
  return score
end

function so(x, uo, memory)
  scorelist = Any[]
  for i = 1:length(memory)
    score = s(x, memory[i] , uo)
    push!(scorelist, score)
  end
  return scorelist
end

function sr(x, ur, dict)
  scorelist = Any[]
  for k in keys(dict)
    score = s(x, dict[k], ur)
    push!(scorelist, score)
  end
  return scorelist
end
