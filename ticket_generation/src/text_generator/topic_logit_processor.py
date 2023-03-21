import torch
from transformers import LogitsProcessor


class TopicLogitsProcessor(LogitsProcessor):
    def __init__(self, topic_word_vector: torch.Tensor, gamma: int, logit_threshold: int, *args, **kwargs):
        """
        Code taken and adapted from https://github.com/roholazandie/topic_language_generation
        Paper link: https://arxiv.org/abs/2103.06434

        Args:
            topic_word_vector (torch.Tensor): matrix KxV, where K is the number of topics and V the vocabulary size
            gamma (int): strength of topic generation, higher values of γ result in more on-topic text generation
            logit_threshold (int): lower values of threshold correlates with more on-topic text generation because
                                   we change more tokens from the original model
        """

        super().__init__(*args, **kwargs)
        self.topic_word_vector = topic_word_vector
        self.gamma = gamma
        self.logit_threshold = logit_threshold

    def __call__(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Example ( V(vocabulary size) =4 )

        logits:                 [ -600, -700, -60, -50 ] # Output of the model
        indices = logits < LOGIT_THRESHOLD
        indices:                [ 1   , 1   , 0  , 0   ] # Calculated from logits using threshold

        topic_word_vector:      [ 0.498, 0.001, 0.498, 0.001 ] # Topic vector, input of the program
        logscores = torch.log(self.topic_word_vector)
        logscores:              [ -0.3, -inf, -0.3, -inf ]     # log(topic_word_vector)

        -->
        logscores[indices] = logits[indices]
        logscores:              [ -600, -700, -0.3, -inf ] # The only values that preserv an high value are the one
                                                           # that have an high "probability" for both the model
                                                           # and the topic vector

        total_logit = logits + gamma * logscores
        """

        gamma = self.gamma  # higher values of gamma corresponds to more on topic
        LOGIT_THRESHOLD = self.logit_threshold  # smaller values of Threshold is more on topic

        logits = logits.squeeze(0)

        # Get log, so tokens with low values will have big negative values and tokens with
        # high values will have values nearer to 0
        logscores = torch.log(self.topic_word_vector)

        # Get indices of "not correlated" words
        indices = logits < LOGIT_THRESHOLD  # TODO cut logits relatively not absolutely

        # self.double() is equivalent to self.to(torch.float64)
        # update the log(topic_word_vector): the words which were highly unlikely according to the
        # model ( < Threshold ) sre set equal to the logits calculated by the model
        # Therefore in the end, in logscores the only "high" values will be of the words correlated to the
        # topic and likely for the model
        logscores[indices] = logits[indices].double()

        # Update the logits with the logscores
        total_logit = logits + gamma * logscores

        total_logit[torch.isnan(total_logit)] = -float("inf")

        return total_logit.unsqueeze(0)
