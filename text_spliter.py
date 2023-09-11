
class Splitter:
    separators = '.?!;,: '

    @staticmethod
    def split(text, limit):
        return Splitter.__split(text, limit) or [text]

    @staticmethod
    def __split(text, limit):
        for separator in Splitter.separators:
            parts = Splitter.__split_by_separator(text, limit, separator)
            if parts is not None:

                index = 0
                while index < len(parts):
                    part = parts[index]
                    if len(part) > limit:
                        parts_of_part = Splitter.__split(part, limit)
                        if parts_of_part is not None:
                            del parts[index]
                            parts[index:index] = parts_of_part
                            index += len(parts_of_part)
                            continue
                    index += 1

                return parts
        return None

    @staticmethod
    def __split_by_separator(text, limit, separator):
        words = [w.strip() for w in text.split(separator)]
        if len(words) <= 1:
            return None
        #if max(map(len, words)) > limit:
        #    return []
        res, part, others = [], words[0], words[1:]
        for word in others:
            if len(separator) + len(word) > limit - len(part):
                res.append(part)
                part = word
            else:
                part += separator + word
        if part:
            res.append(part)
        return res


