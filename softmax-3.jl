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
  vocabDict = createDict([training_data, test_data])
  vocabDict_length = length(vocabDict)
  info("$vocabDict_length unique words.")
  feature_space = 5 * vocabDict_length

  model = initWeights(settings[:atype], feature_space, embedding_dimension, settings[:winit])
  uo_adam = Adam(;lr = uo_lr)
  ur_adam = Adam(;lr = ur_lr)

  for epoch = 0:total_epochs
    @time o_avg_loss, r_avg_loss = train(training_data, model[:uo], model[:ur], vocabDict, uo_adam, ur_adam, settings[:atype], epoch)
    o_accuracy, r_accuracy = trainingAccuracy(training_data, model[:uo], model[:ur], vocabDict, settings[:atype])
    @time test_o_avg_loss, test_r_avg_loss, test_accuracy = test(test_data, model[:uo], model[:ur], vocabDict, settings[:atype])
    println("Epoch: $epoch => [o_loss: $o_avg_loss, r_loss: $r_avg_loss, o_accuracy: $o_accuracy %, r_accuracy: $r_accuracy %], [o_loss: $test_o_avg_loss, r_loss: $test_r_avg_loss, accuracy: $test_accuracy %]")
  end
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
    for i = 2:length(words) - 3
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

function O(x_feature_rep, memory, q_list, uo, atype)
  x_feature_rep_list1 = [x_feature_rep]
  scoreArray1 = so(x_feature_rep_list1, memory, q_list, uo, atype)
  scoreArray1 = scoreArray1 .- maximum(scoreArray1, 1)
  scoreProb1 = exp(scoreArray1) ./ sum(exp(scoreArray1), 1)
  o1 = indmax(scoreProb1)
  mo1 = memory[o1]

  x_feature_rep_list2 = [x_feature_rep, mo1]
  scoreArray2 = so(x_feature_rep_list2, memory, q_list, uo, atype)
  scoreArray2 = scoreArray2 .- maximum(scoreArray2, 1)
  scoreProb2 = exp(scoreArray2) ./ sum(exp(scoreArray2), 1)
  o2 = indmax(scoreProb2)
  mo2 = memory[o2]

  x_feature_rep_list3 = [x_feature_rep, mo1, mo2]
  scoreArray3 = so(x_feature_rep_list3, memory, q_list, uo, atype)
  scoreArray3 = scoreArray3 .- maximum(scoreArray3, 1)
  scoreProb3 = exp(scoreArray3) ./ sum(exp(scoreArray3), 1)
  o3 = indmax(scoreProb3)
  mo3 = memory[o3]

  return [x_feature_rep, mo1, mo2, mo3]
end

function phix(feature_rep_list, atype)
  mapped = copy(feature_rep_list[1])
  for i = 2:5
    if i <= length(feature_rep_list)
      mapped = vcat(mapped, feature_rep_list[i])
    else
      mapped = vcat(mapped, zeros(Float64, length(feature_rep_list[1]), 1))
    end
  end
  mapped = convert(atype, mapped)
  return mapped
end

function phiy(feature_rep, atype)
  mapped = copy(feature_rep)
  mapped = vcat(zeros(Float64, 4 * length(feature_rep), 1), mapped)
  mapped = convert(atype, mapped)
  return mapped
end

function s(x_feature_rep_list, y_feature_rep, u, atype)
  phi_y = phiy(y_feature_rep, atype)
  phi_x = phix(x_feature_rep_list, atype)
  score = sum(phi_x' * u' * u * phi_y)
  return score
end

function so(x_feature_rep_list, memory, q_list, uo, atype)
  scoreArray = s(x_feature_rep_list, memory[1], uo, atype)
  for i = 2:length(memory)
    if in(i, q_list)
      score = -Inf
    else
      score = s(x_feature_rep_list, memory[i], uo, atype)
    end
    scoreArray = vcat(scoreArray, score)
  end
  return scoreArray
end

function R(input_list, vocabDict, ur, atype)
  scoreArray = sr(input_list, vocabDict, ur, atype)
  scoreArray = scoreArray .- maximum(scoreArray, 1)
  scoreProb = exp(scoreArray) ./ sum(exp(scoreArray), 1)
  index = indmax(scoreArray)
  for k in keys(vocabDict)
    if vocabDict[k] == index
      return k
    end
  end
end

function sr(x_feature_rep_list, vocabDict, ur, atype)
  w = findWord(vocabDict, 1)
  y_feature_rep = word2OneHot(w, vocabDict)
  scoreArray = s(x_feature_rep_list, y_feature_rep, ur, atype)
  for i = 2:length(vocabDict)
    w = findWord(vocabDict, i)
    y_feature_rep = word2OneHot(w, vocabDict)
    score = s(x_feature_rep_list, y_feature_rep, ur, atype)
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

function uoSoftloss(uo, x_feature_rep_list1, memory, q_list, vocabDict, golds, atype)
  uoArray1 = so(x_feature_rep_list1, memory, q_list, uo, atype)
  uoArray1 = uoArray1 .- maximum(uoArray1, 1)
  uoProb1 = exp(uoArray1) ./ sum(exp(uoArray1), 1)
  o1 = indmax(uoProb1)
  mo1 = memory[o1]
  uoLoss = (-1) * log(uoProb1[golds[1]])

  x_feature_rep_list2 = [x_feature_rep_list1[1], mo1]
  uoArray2 = so(x_feature_rep_list2, memory, q_list, uo, atype)
  uoArray2 = uoArray2 .- maximum(uoArray2, 1)
  uoProb2 = exp(uoArray2) ./ sum(exp(uoArray2), 1)
  o2 = indmax(uoProb2)
  mo2 = memory[o2]
  uoLoss = uoLoss + (-1) * log(uoProb2[golds[2]])

  x_feature_rep_list3 = [x_feature_rep_list1[1], mo1, mo2]
  uoArray3 = so(x_feature_rep_list3, memory, q_list, uo, atype)
  uoArray3 = uoArray3 .- maximum(uoArray3, 1)
  uoProb3 = exp(uoArray3) ./ sum(exp(uoArray3), 1)
  uoLoss = uoLoss + (-1) * log(uoProb3[golds[3]])

  return uoLoss
end

uoSoftlossGrad = grad(uoSoftloss)

function urSoftloss(ur, x_feature_rep_list, memory, vocabDict, gold, atype)
  urArray = sr(x_feature_rep_list, vocabDict, ur, atype)
  urArray = urArray .- maximum(urArray, 1)
  urProb = exp(urArray) ./ sum(exp(urArray), 1)
  urLoss = (-1) * log(urProb[vocabDict[gold]])
  return urLoss
end

urSoftlossGrad = grad(urSoftloss)

function train(data_file, uo, ur, vocabDict, uo_adam, ur_adam, atype, epoch)
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
      for i = 3:(length(words) - 4)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        question = question * " " * words[i]
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - 3]

      correct_m1_index = words[end - 2]
      correct_m1_index = parse(Int, correct_m1_index)
      correct_m1 = memory[correct_m1_index]

      correct_m2_index = words[end - 1]
      correct_m2_index = parse(Int, correct_m2_index)
      correct_m2 = memory[correct_m2_index]

      correct_m3_index = words[end]
      correct_m3_index = parse(Int, correct_m3_index)
      correct_m3 = memory[correct_m3_index]

      uoGolds = [correct_m1_index, correct_m2_index, correct_m3_index]
      uoLoss = uoSoftloss(uo, [question_feature_rep], memory, q_list, vocabDict, uoGolds, atype)
      uoLossGradient = uoSoftlossGrad(uo, [question_feature_rep], memory, q_list, vocabDict, uoGolds, atype)

      urLoss = urSoftloss(ur, [question_feature_rep, correct_m1, correct_m2, correct_m3], memory, vocabDict, correct_r, atype)
      urLossGradient = urSoftlossGrad(ur, [question_feature_rep, correct_m1, correct_m2, correct_m3], memory, vocabDict, correct_r, atype)

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

function trainingAccuracy(data_file, uo, ur, vocabDict, atype)
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
      for i = 3:(length(words) - 4)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        question = question * " " * words[i]
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - 3]

      correct_m1_index = words[end - 2]
      correct_m1_index = parse(Int, correct_m1_index)
      correct_m1 = memory[correct_m1_index]

      correct_m2_index = words[end - 1]
      correct_m2_index = parse(Int, correct_m2_index)
      correct_m2 = memory[correct_m2_index]

      correct_m3_index = words[end]
      correct_m3_index = parse(Int, correct_m3_index)
      correct_m3 = memory[correct_m3_index]

      output = O(question_feature_rep, memory, q_list, uo, atype)
      if in(correct_m1, output) && in(correct_m2, output) && in(correct_m3, output)
        numsup = numsup + 1
      end

      response = R([question_feature_rep, correct_m1, correct_m2, correct_m3], vocabDict, ur, atype)
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

function test(data_file, uo, ur, vocabDict, atype)
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
      for i = 3:(length(words) - 4)
        if words[i][end] == '?' || words[i][end] == '.'
          words[i] = words[i][1:end - 1]
        end
        question = question * " " * words[i]
      end
      question_feature_rep = I(question, vocabDict, atype)
      G(question_feature_rep, memory)
      push!(q_list, line_number)

      correct_r = words[end - 3]

      correct_m1_index = words[end - 2]
      correct_m1_index = parse(Int, correct_m1_index)
      correct_m1 = memory[correct_m1_index]

      correct_m2_index = words[end - 1]
      correct_m2_index = parse(Int, correct_m2_index)
      correct_m2 = memory[correct_m2_index]

      correct_m3_index = words[end]
      correct_m3_index = parse(Int, correct_m3_index)
      correct_m3 = memory[correct_m3_index]

      uoGolds = [correct_m1_index, correct_m2_index, correct_m3_index]
      uoLoss = uoSoftloss(uo, [question_feature_rep], memory, q_list, vocabDict, uoGolds, atype)

      urLoss = urSoftloss(ur, [question_feature_rep, correct_m1, correct_m2, correct_m3], memory, vocabDict, correct_r, atype)

      response = answer(question_feature_rep, memory, q_list, vocabDict, uo, ur, atype)
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

function answer(x_feature_rep, memory, q_list, vocabDict, uo, ur, atype)
  output = O(x_feature_rep, memory, q_list, uo, atype)
  answer = R(output, vocabDict, ur, atype)
  return answer
end

main()
