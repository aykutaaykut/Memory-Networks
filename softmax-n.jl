using Knet, AutoGrad, Compat, ArgParse

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    ("--datafiles"; nargs = '+'; help = "If provided, use first file for training, second for test.")
    ("--winit"; arg_type=Float64; default=0.01; help="Initial weights set to winit*randn().")
    ("--seed"; arg_type=Int; default=38; help="Random number seed.")
    ("--atype"; default = (gpu() >= 0 ? "KnetArray{Float64}" : "Array{Float64}"); help = "Array type: Array for CPU, KnetArray for GPU")
  end
  return parse_args(s; as_symbols = true)
end

function main(args = ARGS)
  #Configurations of the model
  embedding_dimension = 100
  uo_lr = 0.001
  ur_lr = 0.001
  total_epochs = 100

  #User settings are added.
  settings = parse_commandline()
  println("settings", [(symbol, value) for (symbol, value) in settings]...)
  settings[:seed] > 0 && srand(settings[:seed])
  settings[:atype] = eval(parse(settings[:atype]))

  #Classification of datafiles as training and test.
  training_data = settings[:datafiles][1]
  if length(settings[:datafiles]) > 1
    test_data = settings[:datafiles][2]
  end

  #Creating the dictionary of words in the data.
  d_factor = findMaxSupFact([training_data, test_data])
  vocabDict = createDict([training_data, test_data])
  vocabDict_length = length(vocabDict)
  info("$vocabDict_length unique words.")
  feature_space = d_factor * vocabDict_length

  model = initWeights(settings[:atype], feature_space, embedding_dimension, settings[:winit])
  uo_adam = Adam(;lr = uo_lr)
  ur_adam = Adam(;lr = ur_lr)

  for epoch = 0:total_epochs
    @time o_avg_loss, r_avg_loss = train(training_data, model[:uo], model[:ur], vocabDict, uo_adam, ur_adam, d_factor, settings[:atype], epoch)
    o_accuracy, r_accuracy = trainingAccuracy(training_data, model[:uo], model[:ur], vocabDict, d_factor, settings[:atype])
    @time test_o_avg_loss, test_r_avg_loss, test_accuracy = test(test_data, model[:uo], model[:ur], vocabDict, d_factor, settings[:atype])
    println("Epoch: $epoch => [o_loss: $o_avg_loss, r_loss: $r_avg_loss, o_accuracy: $o_accuracy %, r_accuracy: $r_accuracy %], [o_loss: $test_o_avg_loss, r_loss: $test_r_avg_loss, accuracy: $test_accuracy %]")
  end
end

function findMaxSupFact(datafiles)
  maxSupCount = 0
  for file in datafiles
    f = open(file)
    while !eof(file)
      str = readline(f)
      words = split(str)
      if words[end][end] != '.'
        count = 0
        for w in words
          if isnumber(w)
            count = count + 1
          end
        end
        count = count - 1 #line number is not counted as a supporting fact.
      end
      if count > maxSupCount
        maxSupCount = count
      end
    end
  end
  return maxSupCount
end

function createDict(datafiles)
  dict = Dict{String, Int}()
  number = 1
  for file in datafiles
    f = open(file)
    while !eof(f)
      str = readline(f)
      number, dict = parseLineAddDict(number, str, dict)
    end
    close(f)
  end
  return dict
end

function parseLineAddDict(number, line, dict)
  words = split(line)
  if words[end][end] == '.'
    for i = 2:length(words)
      str = words[i]
      if str[end] == '.'
        str = str[1:end - 1]
      end
      if !haskey(dict, str)
        dict[str] = number
        number = number + 1
      end
    end
  else
    for i = 2:length(words) - 2
      str = words[i]
      if str[end] == '?'
        str = str[1:end - 1]
      end
      if !haskey(dict, str)
        dict[str] = number
        number = number + 1
      end
    end
  end
  return number, dict
end

function initWeights(atype, feature_space, embedding_dimension, winit)
  weights = Dict{Symbol, Any}()

  uo = winit * randn(embedding_dimension, feature_space)
  ur = winit * randn(embedding_dimension, feature_space)
  weights[:uo] = uo
  weights[:ur] = ur

  for k in keys(weights)
    weights[k] = convert(atype, weights[k])
  end
  return weights
end

function I(x, dict, atype)
  words = split(x)
  feature_rep = zeros(Float64, length(dict), 1)
  for w in words
    if w[end] == '?' || w[end] == '.'
      w = w[1:end - 1]
    end
    onehot = word2OneHot(w, dict)
    feature_rep = feature_rep .+ onehot
  end
  feature_rep = convert(atype, feature_rep)
  return feature_rep
end

function word2OneHot(word, dict)
  onehot = zeros(Float64, length(dict), 1)
  for w in keys(dict)
    if w == word
      onehot[dict[w], 1] = 1.0
      break
    end
  end
  return onehot
end

function G(feature_rep, memory)
  push!(memory, feature_rep)
  return memory
end

function O(x_feature_rep, memory, q_list, uo, d_factor, atype)
  x_feature_rep_list = [x_feature_rep]
  for i = 1:d_factor
    scoreArray = so(x_feature_rep_list, memory, q_list, uo, d_factor, atype)
    scoreArray = scoreArray .- maximum(scoreArray, 1)
    scoreProb = exp(scoreArray) ./ sum(exp(scoreArray), 1)
    o = indmax(scoreProb)
    mo = memory[o]
    temp_mo = [mo]
    x_feature_rep_list = vcat(x_feature_rep_list, temp_mo)
  end

  return x_feature_rep_list
end

function phix(feature_rep_list, d_factor, atype)
  mapped = copy(feature_rep_list[1])
  for i = 2:d_factor
    if i <= length(feature_rep_list)
      mapped = vcat(mapped, feature_rep_list[i])
    else
      mapped = vcat(mapped, zeros(Float64, length(feature_rep_list[1]), 1))
    end
  end
  mapped = convert(atype, mapped)
  return mapped
end

function phiy(feature_rep, d_factor, atype)
  mapped = copy(feature_rep)
  mapped = vcat(zeros(Float64, (d_factor - 1) * length(feature_rep), 1), mapped)
  mapped = convert(atype, mapped)
  return mapped
end

function s(x_feature_rep_list, y_feature_rep, u, d_factor, atype)
  phi_y = phiy(y_feature_rep, d_factor, atype)
  phi_x = phix(x_feature_rep_list, d_factor, atype)
  score = sum(phi_x' * u' * u * phi_y)
  return score
end

function so(x_feature_rep_list, memory, q_list, uo, d_factor, atype)
  scoreArray = s(x_feature_rep_list, memory[1], uo, d_factor, atype)
  for i = 2:length(memory)
    if in(i, q_list)
      score = -Inf
    else
      score = s(x_feature_rep_list, memory[i], uo, d_factor, atype)
    end
    scoreArray = vcat(scoreArray, score)
  end
  return scoreArray
end

function R(input_list, vocabDict, ur, d_factor, atype)
  scoreArray = sr(input_list, vocabDict, ur, d_factor, atype)
  scoreArray = scoreArray .- maximum(scoreArray, 1)
  scoreProb = exp(scoreArray) ./ sum(exp(scoreArray), 1)
  index = indmax(scoreArray)
  for k in keys(vocabDict)
    if vocabDict[k] == index
      return k
    end
  end
end

function sr(x_feature_rep_list, vocabDict, ur, d_factor, atype)
  w = findWord(vocabDict, 1)
  y_feature_rep = word2OneHot(w, vocabDict)
  scoreArray = s(x_feature_rep_list, y_feature_rep, ur, d_factor, atype)
  for i = 2:length(vocabDict)
    w = findWord(vocabDict, i)
    y_feature_rep = word2OneHot(w, vocabDict)
    score = s(x_feature_rep_list, y_feature_rep, ur, d_factor, atype)
    scoreArray = vcat(scoreArray, score)
  end
  return scoreArray
end

function findWord(vocabDict, i)
  for k in keys(vocabDict)
    if vocabDict[k] == i
      return k
    end
  end
end

function uoSoftloss(uo, x_feature_rep_list, memory, q_list, vocabDict, golds, d_factor, atype)
  uoLoss = 0
  for i = 1:d_factor
    uoArray = so(x_feature_rep_list, memory, q_list, uo, d_factor, atype)
    uoArray = uoArray .- maximum(uoArray, 1)
    uoProb = exp(uoArray) ./ sum(exp(uoArray), 1)
    o = indmax(uoProb)
    mo = memory[o]
    temp_mo = [mo]
    x_feature_rep_list = vcat(x_feature_rep_list, temp_mo)
    uoLoss = uoLoss + (-1) * log(uoProb[golds[i]])
  end

  return uoLoss
end

uoSoftlossGrad = grad(uoSoftloss)

function urSoftloss(ur, x_feature_rep_list, memory, vocabDict, gold, d_factor, atype)
  urArray = sr(x_feature_rep_list, vocabDict, ur, d_factor, atype)
  urArray = urArray .- maximum(urArray, 1)
  urProb = exp(urArray) ./ sum(exp(urArray), 1)
  urLoss = (-1) * log(urProb[vocabDict[gold]])
  return urLoss
end

urSoftlossGrad = grad(urSoftloss)

function train(data_file, uo, ur, vocabDict, uo_adam, ur_adam, d_factor, atype, epoch)
  uo_total_loss = 0
  ur_total_loss = 0
  numq = 0
  memory = resetMemory()
  q_list = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      line_number = parse(Int, line_number)
      sentence = words[2]
      for i = 3:length(words)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
        q_list = resetMemory()
      end
      sentence_feature_rep = I(sentence, vocabDict, atype)
      G(sentence_feature_rep, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      sup_res = 0
      for i = 3:length(words)
        if words[i][end] =='?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        if !isnumber(words[min(i + 1, length(words))])
          question = question * " " * words[i]
        else
          sup_res = sup_res + 1
        end
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - sup_res + 1]

      correct_ms = Any[]
      correct_ms_index = Any[]
      for i = (length(words) - sup_res + 2):length(words)
        correct_m_index = words[i]
        correct_m_index = parse(Int, correct_m_index)
        correct_m = memory[correct_m_index]
        push!(correct_ms, correct_m)
        push!(correct_ms_index, correct_m_index)
      end

      uoLoss = uoSoftloss(uo, [question_feature_rep], memory, q_list, vocabDict, correct_ms_index, d_factor, atype)
      uoLossGradient = uoSoftlossGrad(uo, [question_feature_rep], memory, q_list, vocabDict, correct_ms_index, d_factor, atype)

      urInput = [question_feature_rep]
      urInput = vcat(urInput, correct_ms)
      urLoss = urSoftloss(ur, urInput, memory, vocabDict, correct_r, d_factor, atype)
      urLossGradient = urSoftlossGrad(ur, urInput, memory, vocabDict, correct_r, d_factor, atype)

      if epoch != 0
        update!(uo, uoLossGradient, uo_adam)
        update!(ur, urLossGradient, ur_adam)
      end

      uo_total_loss = uo_total_loss + uoLoss
      ur_total_loss = ur_total_loss + urLoss
      numq = numq + 1
    end
  end
  close(f)
  uo_avg_loss = uo_total_loss / numq
  ur_avg_loss = ur_total_loss / numq
  return uo_avg_loss, ur_avg_loss
end

function resetMemory()
  return Any[]
end

function trainingAccuracy(data_file, uo, ur, vocabDict, d_factor, atype)
  numsup = 0
  numcorr = 0
  numq = 0
  memory = resetMemory()
  q_list = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      line_number = parse(Int, line_number)
      sentence = words[2]
      for i = 3:length(words)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
        q_list = resetMemory()
      end
      sentence_feature_rep = I(sentence, vocabDict, atype)
      G(sentence_feature_rep, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      sup_res = 0
      for i = 3:length(words)
        if words[i][end] =='?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        if !isnumber(words[min(i + 1, length(words))])
          question = question * " " * words[i]
        else
          sup_res = sup_res + 1
        end
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - sup_res + 1]

      correct_ms = Any[]
      correct_ms_index = Any[]
      for i = (length(words) - sup_res + 2):length(words)
        correct_m_index = words[i]
        correct_m_index = parse(Int, correct_m_index)
        correct_m = memory[correct_m_index]
        push!(correct_ms, correct_m)
        push!(correct_ms_index, correct_m_index)
      end

      output = O(question_feature_rep, memory, q_list, uo, d_factor, atype)
      corr = true
      for i = 1:length(correct_ms)
        if !in(correct_ms[i], output)
          corr = false
        end
      end
      if corr
        numsup = numsup + 1
      end

      RInput = [question_feature_rep]
      RInput = vcat(RInput, correct_ms)
      response = R(RInput, vocabDict, ur, d_factor, atype)
      if response == correct_r
        numcorr = numcorr + 1
      end

      numq = numq + 1
    end
  end
  close(f)
  output_accuracy = numsup / numq * 100
  response_accuracy = numcorr / numq * 100
  return output_accuracy, response_accuracy
end

function test(data_file, uo, ur, vocabDict, d_factor, atype)
  numcorr = 0
  uo_total_loss = 0
  ur_total_loss = 0
  numq = 0
  memory = resetMemory()
  q_list = resetMemory()
  f = open(data_file)
  while !eof(f)
    str = readline(f)
    words = split(str)
    if words[end][end] == '.'
      line_number = words[1]
      line_number = parse(Int, line_number)
      sentence = words[2]
      for i = 3:length(words)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        sentence = sentence * " " * words[i]
      end
      if line_number == 1
        memory = resetMemory()
        q_list = resetMemory()
      end
      sentence_feature_rep = I(sentence, vocabDict, atype)
      G(sentence_feature_rep, memory)
    else
      line_number = words[1]
      line_number = parse(Int, line_number)
      question = words[2]
      sup_res = 0
      for i = 3:length(words)
        if words[i][end] =='?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        if !isnumber(words[min(i + 1, length(words))])
          question = question * " " * words[i]
        else
          sup_res = sup_res + 1
        end
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - sup_res + 1]

      correct_ms = Any[]
      correct_ms_index = Any[]
      for i = (length(words) - sup_res + 2):length(words)
        correct_m_index = words[i]
        correct_m_index = parse(Int, correct_m_index)
        correct_m = memory[correct_m_index]
        push!(correct_ms, correct_m)
        push!(correct_ms_index, correct_m_index)
      end

      uoLoss = uoSoftloss(uo, [question_feature_rep], memory, q_list, vocabDict, correct_ms_index, atype)

      urInput = [question_feature_rep]
      urInput = vcat(urInput, correct_ms)
      urLoss = urSoftloss(ur, urInput, memory, vocabDict, correct_r, d_factor, atype)

      response = answer(question_feature_rep, memory, q_list, vocabDict, uo, ur, d_factor, atype)
      if response == correct_r
        numcorr = numcorr + 1
      end

      uo_total_loss = uo_total_loss + uoLoss
      ur_total_loss = ur_total_loss + urLoss
      numq = numq + 1
    end
  end
  close(f)
  test_accuracy = numcorr / numq * 100
  uo_avg_loss = uo_total_loss / numq
  ur_avg_loss = ur_total_loss / numq
  return uo_avg_loss, ur_avg_loss, test_accuracy
end

function answer(x_feature_rep, memory, q_list, vocabDict, uo, ur, d_factor, atype)
  output = O(x_feature_rep, memory, q_list, uo, d_factor, atype)
  answer = R(output, vocabDict, ur, d_factor, atype)
  return answer
end

main()
