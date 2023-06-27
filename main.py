import statistics
import random

class SampleWrapper:
    def __init__(self, sample):
        self.sample = sample

    def get_p_from_0_to_s(self, s):
        counter = 0
        for element in self.sample:
            if element <= s:
                counter += 1
        return counter/len(self.sample)


class PredictionEvaluatorNoContext:
    def __init__(self, val, point, vals):
        self.val = val
        self.point = point
        self.vals = vals

        self.dus_to_pareto_errors = {}
        self.abs_dus_to_samples = {}

        self.init_pareto_errors()
        self.init_dus_to_sample_wrappers()

    #################  MAIN METHOD  #########################
    def get_w(self):
        w = 0
        for du, err in self.dus_to_pareto_errors.items():
            w_du = self.abs_dus_to_samples[abs(du)].get_p_from_0_to_s(s=err)
            w+=w_du
        return w

    #########################################################
    def init_dus_to_sample_wrappers(self):
        for abs_du in range(len(self.vals)):
            sample_du = self.get_sample_for_abs_du(abs_du)
            self.abs_dus_to_samples[abs_du] = SampleWrapper(sample_du)

    def get_sample_for_abs_du(self, abs_du):
        sample_paretos_errs = []

        # выборку какого размера мы хотим собрать для данного abs_du (чем больше, тем лучше)
        sample_len = len(self.vals)

        # случайно набираем vals для использования как баз предсказаний
        random_floats = [random.uniform(0.0, float(max(self.vals))) for _ in range(sample_len)]

        for val_predicted in random_floats:
            # для данного val_predicted замеряем ошибку на всем бассейне vals:
            abs_errs_list = self.get_abs_errs_list(self.vals, val_predicted)

            # семплим из этого списка (без возвращения) abs_du+ 1 штук значений ошибки
            random_sample_errs = random.sample(abs_errs_list, abs_du)
            sample_paretos_errs.append(min(random_sample_errs))

        return sample_paretos_errs

    def get_abs_errs_list(self, vals, predicted_val):
        return list([abs(vals[i]- predicted_val) for i in range(len(vals))])


    def init_pareto_errors(self): # построение мн-ва слейтера на данном бассейне оценки предсказания
        abs_errs_list = self.get_abs_errs_list(self.vals, predicted_val=self.val)
        self.dus_to_pareto_errors[0] = abs_errs_list[self.point]

        # перебираем левее, удаляясь в отрицательные точки от нулевой
        i = self.point
        current_best_err = abs_errs_list[self.point]
        while True:
            new_index = i - 1
            if new_index < 0:
                break
            new_err = abs_errs_list[new_index]
            if new_err < current_best_err:
                current_best_err = new_err
            self.dus_to_pareto_errors[new_index] = current_best_err
            i = new_index

        # перебираем вправо, удаляясь в положительные точки от нулевой
        i = self.point
        current_best_err = abs_errs_list[self.point]
        while True:
            new_index = i + 1
            if new_index == len(abs_errs_list):
                break
            new_err = abs_errs_list[new_index]
            if new_err < current_best_err:
                current_best_err = new_err
            self.dus_to_pareto_errors[new_index] = current_best_err
            i = new_index


def eval_prediction_in_contexted_mean(val, point, vals):
    evaluator_new_pred = PredictionEvaluatorNoContext(val, point, vals)
    mean_val = statistics.mean(vals)
    evaluator_def_pred = PredictionEvaluatorNoContext(mean_val, point, vals)
    w = evaluator_new_pred.get_w() - evaluator_def_pred.get_w()
    return w


if __name__ == '__main__':
    vals = [1, 3, 3, 3]
    point = 0
    val = 0.8

    w = eval_prediction_in_contexted_mean(val, point, vals)

    print("    w = " + str(w))

