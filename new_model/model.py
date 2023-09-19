from typing import List

import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig



config = AutoConfig.from_pretrained('cointegrated/rubert-tiny2', dropout=0.25, attention_dropout=0.25)
bert = AutoModel.from_pretrained('cointegrated/rubert-tiny2', config=config)
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')

MAX_COMMENT_LEN = 400


class_mapper = [
    "General positive - blank",
    "General positive - service",
    "General comments - product",
    "Smell - usage",
    "Comparison with cigarettes - usage",
    "Ease of use",
    "Smoking and nicotine harm - recommendations",
    "General positive - sales expert",
    "Already recommended - recommendations",
    "Device breakage - product",
    "Taste",
    "Completely switched to RRP - usage",
    "Design - product",
    "No ash, smoke, combustion - usage",
    "Sticks quality - product",
    "Improved health - health",
    "Comments on score - recommendations",
    "Replacement - service",
    "Sticks price",
    "Device battery - product",
    "Sticks assortment - service",
    "General price",
    "Session duration - usage",
    "Discounts/ promotion",
    "General negative - service",
    "Cleaning - usage",
    "Expertise - sales expert",
    "Quit nicotine completely",
    "Comparing device versions - product",
    "Problem not solved - service",
    "General negative - blank",
    "Social inclusion",
    "Can use everywhere - usage",
    "Everyone chooses himself - recommendations",
    "Device comparison with competitors - product",
    "No saturation - usage",
    "Adverse event - health",
    "Early poll - recommendations",
    "Charger and holder price",
    "General positive - remote care",
    "Delivery speed",
    "Company and brand - blank",
    "Improvement idea - product",
    "Order & delivery details accuracy",
    "Trial application process",
    "Store location and coverage",
    "Comparison with electronic cigarettes - product",
    "Problem not solved - remote care",
    "Spam - communication",
    "Accessories quality - product",
    "Indulgence",
    "Store atmosphere (e.g. light, amenities)",
    "Rudeness - sales expert",
    "Charger and holder assortment - service",
    "Problem not solved - sales expert",
    "Long service - sales expert",
    "No friends - recommendations",
    "Accessories assortment - service",
    "Wrong time - delivery",
    "Long service - remote care",
    "Sticks availability - service",
    "Loyalty points accumulation and redemption",
    "Store cleanliness/ tidiness",
    "Courier service - delivery",
    "Cancellation not by client will - delivery",
    "Device LE - product",
    "Fraud - sales expert",
    "Personal data change - communication",
    "Charger and holder availability - service",
    "Out delivery area - delivery",
    "Different communication - service",
    "Lending - service",
    "Device indication - product",
    "Technical issue - service",
    "Expertise - remote care",
    "Technical problems - website",
    "Checkout and payment process",
    "Return - service",
    "Accessories price",
    "Queue - sales expert",
    "Marketing content & design",
    "Service refused - sales expert",
    "Accessories availability - service",
    "Service refused - service",
    "Broke device - sales expert",
    "No information - website",
    "Rudeness - remote care",
    "Registration process",
    "Website design",
    "Store closed - sales expert",
    "Store locator - website",
    "Bot - service",
    "Blank"
]

class ReplyModel(pl.LightningModule):

    def __init__(self, num_labels: int = 92):
        super().__init__()

        self.save_hyperparameters()
        self.sched = None

        self.num_labels = num_labels
        self.bert = bert

        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(0.35)
        self.relu = torch.nn.ReLU()


    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_outputs(input_ids, attention_mask)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits

    def get_outputs(self, input_ids, attention_mask):
        outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
        )
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        return pooled_output

    def get_categories(self, comments: List[str]) -> List[List[str]]:
        encoding = tokenizer.batch_encode_plus(
          comments,
          add_special_tokens=True,
          max_length=MAX_COMMENT_LEN,
          #return_token_type_ids=True,
          truncation=True,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )
        model_output = torch.sigmoid(
                self(encoding['input_ids'], encoding['attention_mask'])
        )
        cat_ids: torch.Tensor = model_output > 0.85
        cat_ids = cat_ids.nonzero()
        result = []
        for i in range(len(comments)):
            categories = [int(cat_id) for k, cat_id in cat_ids if k == i]
            if not categories:
                categories = torch.topk(model_output[i], 1)
                if categories.values[0] < 0.5:
                    result.append(class_mapper[93])
                    continue
                categories = [categories.indices[0]]
            categories.sort(key=lambda x: model_output[i, x], reverse=True)
            result.append([class_mapper[cat] for cat in categories])
        return result

if __name__ == '__main__':
    model = ReplyModel().load_from_checkpoint(
            "epoch=79-step=12720_28082023.ckpt",
            map_location=torch.device('cpu'),
            strict=False
    )
    model.eval()
    model.freeze()
    a = model.get_categories(
            [

"Пусть сами решают",
"Почему так нравится мне просто это ж ваше приспособление",
"Потому что вы хороши",
"Спасибо за вашу продукцию.Если для окружающих стики не будут сильно пахнуть, и нам, курящим будет лучше ощущаться "
"вкус, цены этому прекрасному изобретению не будет.Почти все друзья бросили с ним курить сигареты:)",
"Потому что не решили мою проблему",
"Это лучше сигарет",
"Айкос удобен в использовании,и без запаха",
"В качестве табак не подделка Качество обслуживания Ну что то в этом роде",
"Потому что вопрос по стикам не решился",
"Потому что курить вредно",
"Потому что все хорошо все нравится",
"Такая оценка потому что понравилась консультация потому что отнеслись к моей проблеме с пониманием и все, правильно все понятно разъяснили"
"Стики дорогие",
"Неправильная оценка",
"Не люблю навязывать людям",
"Дружелюбная поддержка",
"Все понравилось",
"Айкос ломается, создавая неудобства(",
"Он лучше чем сигареты",
"Да потому что без вопросов все на высшем уровне",
"Нравится обслуживание",
"Все понравилось все прекрасно все обслуживание качество спасибо",
"А здесь ли вообще не приветливость и желание помочь",
"Курить вредно",
"Видимо потому , что с вами не просто иметь дело , когда обращаешься к вашим специалистом они отказываются выполнять "
"возврат в период гарантии но в последующем они признают ошибку но уже поздно извините гарантия прошла",
"Я пользуюсь мне нравится я хочу чтобы нравилось другим",
"Меньше вреда чем от сигареты",
"Легкий болят",
"проблема не решена",
"Потому что понравилось",
"все понравилось",
"Компания навязывает мне дополнительных услуг в которых я иногда нуждаюсь я прихожу обращаюсь мне очень нравится Мне очень нравится что у вас нет рекламы но вас всех знают",
"Потому что компетентные сотрудники у вас",
"Хорошее обслуживание. Рекомендации друзьям",
"Устройство работает недолго промокоды работают некачественно партнер",
"Меня все устраивает",
"МНЕ НРАВИТСЯ, ЧТО МОИ ВОПРОСЫ МГНОВЕННО РЕШАЮТ",
"Ну потому что клиенты ориентированы",
"Потому что все хорошо😁",
"Курить вредно. Но рас я курю, то я перешёл на айкос.",
"Сотрудника нет на месте, а я очень тороплюсь",
"Потому что идёт введение в заблуждение клиента. Отсутствует заинтересованность в клиенте.",

            ]
    )
    print(a)
