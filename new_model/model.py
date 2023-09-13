from typing import List

import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig



config = AutoConfig.from_pretrained('cointegrated/rubert-tiny2', dropout=0.25, attention_dropout=0.25)
bert = AutoModel.from_pretrained('cointegrated/rubert-tiny2', config=config)
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')

MAX_COMMENT_LEN = 400


class_mapper = [
    "categories1_generalpositiveblank",
    "categories1_generalpositiveservice",
    "categories1_generalcommentsproduct",
    "categories1_smellusage",
    "categories1_comparisonwithcigarettesusage",
    "categories1_easeofuse",
    "categories1_smokingandnicotineharmrecommendations",
    "categories1_generalpositivesalesexpert",
    "categories1_alreadyrecommendedrecommendations",
    "categories1_devicebreakageproduct",
    "categories1_taste",
    "categories1_completelyswitchedtorrpusage",
    "categories1_designproduct",
    "categories1_noashsmokecombustionusage",
    "categories1_sticksqualityproduct",
    "categories1_improvedhealthhealth",
    "categories1_commentsonscorerecommendations",
    "categories1_replacementservice",
    "categories1_sticksprice",
    "categories1_devicebatteryproduct",
    "categories1_sticksassortmentservice",
    "categories1_generalprice",
    "categories1_sessiondurationusage",
    "categories1_discountspromotion",
    "categories1_generalnegativeservice",
    "categories1_cleaningusage",
    "categories1_expertisesalesexpert",
    "categories1_quitnicotinecompletely",
    "categories1_comparingdeviceversionsproduct",
    "categories1_problemnotsolvedservice",
    "categories1_generalnegativeblank",
    "categories1_socialinclusion",
    "categories1_canuseeverywhereusage",
    "categories1_everyonechooseshimselfrecommendations",
    "categories1_devicecomparisonwithcompetitorsproduct",
    "categories1_nosaturationusage",
    "categories1_adverseeventhealth",
    "categories1_earlypollrecommendations",
    "categories1_chargerandholderprice",
    "categories1_generalpositiveremotecare",
    "categories1_deliveryspeed",
    "categories1_companyandbrandblank",
    "categories1_improvementideaproduct",
    "categories1_orderdeliverydetailsaccuracy",
    "categories1_trialapplicationprocess",
    "categories1_storelocationandcoverage",
    "categories1_comparisonwithelectroniccigarettesproduct",
    "categories1_problemnotsolvedremotecare",
    "categories1_spamcommunication",
    "categories1_accessoriesqualityproduct",
    "categories1_indulgence",
    "categories1_storeatmosphereeglightamenities",
    "categories1_rudenesssalesexpert",
    "categories1_chargerandholderassortmentservice",
    "categories1_problemnotsolvedsalesexpert",
    "categories1_longservicesalesexpert",
    "categories1_nofriendsrecommendations",
    "categories1_accessoriesassortmentservice",
    "categories1_wrongtimedelivery",
    "categories1_longserviceremotecare",
    "categories1_sticksavailabilityservice",
    "categories1_loyaltypointsaccumulationandredemption",
    "categories1_storecleanlinesstidiness",
    "categories1_courierservicedelivery",
    "categories1_cancellationnotbyclientwilldelivery",
    "categories1_deviceleproduct",
    "categories1_fraudsalesexpert",
    "categories1_personaldatachangecommunication",
    "categories1_chargerandholderavailabilityservice",
    "categories1_outdeliveryareadelivery",
    "categories1_differentcommunicationservice",
    "categories1_lendingservice",
    "categories1_deviceindicationproduct",
    "categories1_technicalissueservice",
    "categories1_expertiseremotecare",
    "categories1_technicalproblemswebsite",
    "categories1_checkoutandpaymentprocess",
    "categories1_returnservice",
    "categories1_accessoriesprice",
    "categories1_queuesalesexpert",
    "categories1_marketingcontentdesign",
    "categories1_servicerefusedsalesexpert",
    "categories1_accessoriesavailabilityservice",
    "categories1_servicerefusedservice",
    "categories1_brokedevicesalesexpert",
    "categories1_noinformationwebsite",
    "categories1_rudenessremotecare",
    "categories1_registrationprocess",
    "categories1_websitedesign",
    "categories1_storeclosedsalesexpert",
    "categories1_storelocatorwebsite",
    "categories1_botservice",
    "categories1_blank"
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

    def get_categories(self, comments: List[str]):
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
                    result.append(["categories1_blank"])
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
