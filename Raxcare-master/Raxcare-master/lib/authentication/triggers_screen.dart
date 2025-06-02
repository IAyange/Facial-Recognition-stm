import 'package:flutter/material.dart';
import '../../utils/preferences_service.dart';
import '../screens/home/home.dart';
import 'personalization_screen.dart';

class TriggersScreen extends StatefulWidget {
  const TriggersScreen({super.key});

  @override
  State<TriggersScreen> createState() => _TriggersScreenState();
}

class _TriggersScreenState extends State<TriggersScreen> {
  final TextEditingController _triggersController = TextEditingController();
  final Set<String> _selectedTriggers = {};
  bool _isSubmitting = false;

  final List<String> _commonTriggers = [
    'Stress',
    'Anxiety',
    'Social Pressure',
    'Loneliness',
    'Boredom',
    'Certain Places',
    'Specific People',
    'Negative Emotions',
    'Celebrations',
    'Financial Problems'
  ];

  void _toggleTrigger(String trigger) {
    setState(() {
      if (_selectedTriggers.contains(trigger)) {
        _selectedTriggers.remove(trigger);
      } else {
        _selectedTriggers.add(trigger);
      }
    });
  }

  @override
  void dispose() {
    _triggersController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Identify Your Triggers"),
        centerTitle: true,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 32),
            Text(
              "Know Your Triggers",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.primary,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              "What situations or feelings make you vulnerable?",
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey.shade600,
                  ),
            ),
            const SizedBox(height: 24),

            // Common triggers chips
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _commonTriggers.map((trigger) {
                final isSelected = _selectedTriggers.contains(trigger);
                return FilterChip(
                  label: Text(trigger),
                  selected: isSelected,
                  onSelected: (selected) => _toggleTrigger(trigger),
                  selectedColor:
                      Theme.of(context).colorScheme.primary.withOpacity(0.2),
                  checkmarkColor: Theme.of(context).colorScheme.primary,
                  labelStyle: TextStyle(
                    color: isSelected
                        ? Theme.of(context).colorScheme.primary
                        : Colors.grey.shade800,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                    side: BorderSide(
                      color: isSelected
                          ? Theme.of(context).colorScheme.primary
                          : Colors.grey.shade300,
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 24),

            // Custom trigger input
            TextField(
              controller: _triggersController,
              decoration: InputDecoration(
                labelText: "Add your own triggers",
                hintText: "e.g., Work deadlines, Family conflicts",
                prefixIcon: const Icon(Icons.add_circle_outline),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                suffixIcon: _triggersController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.add),
                        onPressed: () {
                          if (_triggersController.text.trim().isNotEmpty) {
                            _toggleTrigger(_triggersController.text.trim());
                            _triggersController.clear();
                          }
                        },
                      )
                    : null,
              ),
              onSubmitted: (value) {
                if (value.trim().isNotEmpty) {
                  _toggleTrigger(value.trim());
                  _triggersController.clear();
                }
              },
            ),
            const Spacer(),

            // Next button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isSubmitting
                    ? null
                    : () async {
                        setState(() => _isSubmitting = true);
                        // Combine selected chips and custom input
                        final allTriggers = _selectedTriggers.toList();
                        if (_triggersController.text.trim().isNotEmpty) {
                          allTriggers.add(_triggersController.text.trim());
                        }

                        await PreferencesService.saveData(
                          'triggers',
                          allTriggers.join(', '),
                        );

                        if (!mounted) return;
                        Navigator.pushReplacement(
                          context,
                          MaterialPageRoute(
                            builder: (context) => PersonalizationScreen(
                              onComplete: () {
                                Navigator.pushReplacement(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => const HomePage(),
                                  ),
                                );
                              },
                            ),
                          ),
                        );
                      },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isSubmitting
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text("Continue"),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
