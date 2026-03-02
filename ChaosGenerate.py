import random
import re
import string
import multiprocessing
from functools import partial

class PromptChaosGenerator:
    def __init__(self, original_text):
        self.original_text = original_text

    @staticmethod
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return PromptChaosGenerator.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def calculate_chaos_score(self, modified_text):
        edit_distance = self.levenshtein_distance(self.original_text.lower(), modified_text.lower())
        
        max_length = max(len(self.original_text), len(modified_text))
        if max_length == 0: return 0
        chaos_score = min(100, (edit_distance / max_length) * 100)
        return round(chaos_score, 2)

    @staticmethod
    def introduce_spelling_errors(text, error_rate=0.1):
        words = text.split()
        modified_words = []
        for word in words:
            if random.random() < error_rate and len(word) > 3:
                error_type = random.choice(['substitute', 'transpose', 'delete', 'insert'])
                
                if error_type == 'substitute' and len(word) > 1:
                    pos = random.randint(0, len(word) - 1)
                    new_char = random.choice(string.ascii_lowercase)
                    word = word[:pos] + new_char + word[pos+1:]
                elif error_type == 'transpose' and len(word) > 2:
                    pos = random.randint(0, len(word) - 2)
                    word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                elif error_type == 'delete' and len(word) > 2:
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos+1:]
                elif error_type == 'insert':
                    pos = random.randint(0, len(word))
                    new_char = random.choice(string.ascii_lowercase)
                    word = word[:pos] + new_char + word[pos:]
            modified_words.append(word)
        return ' '.join(modified_words)

    @staticmethod
    def shuffle_word_order(text, shuffle_rate=0.2):
        words = text.strip().split()
        if len(words) > 3 and random.random() < shuffle_rate:
            num_to_shuffle = max(1, int(len(words) * shuffle_rate))
            for _ in range(num_to_shuffle):
                if len(words) >= 2:
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
        return ' '.join(words)

    @staticmethod
    def remove_random_chars(text, removal_rate=0.05):
        if random.random() < removal_rate:
            chars = list(text)
            num_to_remove = int(len(chars) * removal_rate)
            for _ in range(num_to_remove):
                if chars:
                    pos = random.randint(0, len(chars) - 1)
                    chars.pop(pos)
            return ''.join(chars)
        return text

    def _generate_single_chaos_version(self, target_level):
        best_version_text = self.original_text
        best_score = self.calculate_chaos_score(best_version_text)
        
        for _ in range(50):
            modified_lines = []
            for line in self.original_text.strip().split('\n'):
                parts = line.split(':', 1)
                label, content = parts[0], parts[1]
                
                prompts = content.split(';')
                modified_prompts = []

                for prompt in prompts:
                    version = prompt.strip()
                    if not version: continue

                    error_rate_spell = target_level / 100.0
                    shuffle_rate_word = target_level / 100.0
                    removal_rate_char = target_level / 100.0

                    version = self.introduce_spelling_errors(version, error_rate=error_rate_spell * 0.5)
                    version = self.shuffle_word_order(version, shuffle_rate=shuffle_rate_word * 0.7)
                    version = self.remove_random_chars(version, removal_rate=removal_rate_char * 0.2)
                    
                    modified_prompts.append(version)
                
                modified_lines.append(f"{label}: {'; '.join(modified_prompts)}")

            current_version_text = '\n'.join(modified_lines)
            current_score = self.calculate_chaos_score(current_version_text)
            
            if abs(current_score - target_level) < abs(best_score - target_level):
                best_version_text = current_version_text
                best_score = current_score

        return {
            'text': best_version_text,
            'actual_chaos_score': best_score,
            'target_chaos_score': target_level
        }

    def generate_chaos_versions_parallel(self, chaos_levels=[10, 25, 40, 60, 80]):
        versions = {}
        num_processes = 112
        
        print(f"Starting parallel generation... Using {num_processes} CPU cores.")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(self._generate_single_chaos_version, chaos_levels)
        
        print("All versions generated successfully.")

        for result in results:
            level_key = f"Level_{result['target_chaos_score']}"
            versions[level_key] = result
            
        return versions

if __name__ == '__main__':
    original_prompt = '''  '''
    target_chaos_levels = [5, 15, 30, 50, 75]

    generator = PromptChaosGenerator(original_prompt)
    chaos_versions = generator.generate_chaos_versions_parallel(target_chaos_levels)

    print("\n" + "="*100)
    print("Original:")
    print(original_prompt)
    print("\n" + "="*100 + "\n")

    sorted_levels = sorted(chaos_versions.keys(), key=lambda x: int(x.split('_')[1]))

    for level in sorted_levels:
        data = chaos_versions[level]
        print(f"{level} (Target chaos: {data['target_chaos_score']}, Actual chaos: {data['actual_chaos_score']}):")
        print(data['text'])
        print("\n" + "-"*80 + "\n")

    print("\nChaos Score Guide:")
    print("- 0-10: Minimal errors (occasional typos)")
    print("- 11-25: Light chaos (few spelling errors and word order issues)")
    print("- 26-40: Moderate chaos (noticeable spelling and word order problems)")
    print("- 41-60: High chaos (many errors, but basically readable)")
    print("- 61-80: Severe chaos (heavy errors, difficult to understand)")
    print("- 81-100: Extreme chaos (nearly unrecognizable from original)")