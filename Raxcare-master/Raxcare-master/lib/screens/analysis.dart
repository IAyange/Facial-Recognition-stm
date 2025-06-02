import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:intl/intl.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:shimmer/shimmer.dart';

class RecoveryAnalyticsProvider with ChangeNotifier {
  final GenerativeModel _model;
  SharedPreferences? _prefs;
  String _analysis = ''; // Initialize with empty string instead of null
  DateTime? _lastUpdated;
  bool _isLoading = false;
  Map<String, dynamic>? _userData;

  String get analysis => _analysis; // Changed to non-nullable
  DateTime? get lastUpdated => _lastUpdated;
  bool get isLoading => _isLoading;

  RecoveryAnalyticsProvider({required String apiKey})
      : _model = GenerativeModel(model: 'gemini-1.5-flash', apiKey: apiKey) {
    _init();
  }

  Future<void> _init() async {
    try {
      _prefs = await SharedPreferences.getInstance();
      _userData = await _getUserData(); // Cache user data
      await _loadCachedAnalysis(); // Load previously generated analysis if available
      if (_analysis.isEmpty) {
        await _generateAnalysis(); // Only generate if no cached analysis exists
      }
    } catch (e) {
      _setDefaultAnalysis('Failed to initialize: $e');
    }
  }

  Future<void> _loadCachedAnalysis() async {
    if (_prefs != null) {
      final cachedAnalysis = _prefs!.getString('cached_analysis');
      final cachedTimestamp = _prefs!.getInt('cached_analysis_timestamp');

      if (cachedAnalysis != null && cachedTimestamp != null) {
        _analysis = cachedAnalysis;
        _lastUpdated = DateTime.fromMillisecondsSinceEpoch(cachedTimestamp);
        notifyListeners();
      }
    }
  }

  Future<void> _saveAnalysisToCache() async {
    if (_prefs != null && _analysis.isNotEmpty && _lastUpdated != null) {
      await _prefs!.setString('cached_analysis', _analysis);
      await _prefs!.setInt(
          'cached_analysis_timestamp', _lastUpdated!.millisecondsSinceEpoch);
    }
  }

  Future<void> refreshAnalysis() async {
    await _generateAnalysis();
  }

  void _setDefaultAnalysis(String errorMessage) {
    _analysis = '''
**Analysis Unavailable**  

We couldn't generate your analysis right now.  

Possible reasons:  
- No internet connection  
- Server temporarily unavailable  

Error details: $errorMessage

Please try again later.''';
    _lastUpdated = DateTime.now();
    notifyListeners();
  }

  Future<void> _generateAnalysis() async {
    _isLoading = true;
    notifyListeners();

    try {
      final data = _userData ?? await _getUserData();
      final prompt = '''
Analyze this recovery progress (BE CONCISE BUT INSIGHTFUL):

**Recovery Data:**
- Days in recovery: ${data['days']}
- Current stage: ${data['stage']}
- Addiction type: ${data['type']}
- Support system: ${data['hasSupport'] ? 'Yes' : 'No'}

**Requested Analysis:**
1. Progress summary (1-2 sentences)
2. Key strengths (bullet points)
3. Recommended next steps (bullet points)
4. Motivational quote (optional)

Use markdown formatting with bold headers for each section.''';

      final response = await _model.generateContent([Content.text(prompt)]);
      if (response.text != null && response.text!.isNotEmpty) {
        _analysis = response.text!;
        _lastUpdated = DateTime.now();
        await _saveAnalysisToCache(); // Save to cache after successful generation
      } else {
        _setDefaultAnalysis('Empty response received');
      }
    } catch (e) {
      _setDefaultAnalysis(e.toString());
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>> _getUserData() async {
    if (_userData != null) {
      return _userData!; // Return cached data if available
    }

    if (_prefs == null) {
      return {
        'days': 0,
        'stage': 'Beginning',
        'type': 'General',
        'hasSupport': false,
      };
    }

    final supportString = _prefs!.getString('support_system');
    final hasSupport = supportString == 'true';

    final data = {
      'days': _prefs!.getInt('total_check_ins') ?? 0,
      'stage': _prefs!.getString('recovery_stage') ?? 'Beginning',
      'type': _prefs!.getString('addiction_type') ?? 'General',
      'hasSupport': hasSupport,
    };

    _userData = data; // Cache the data
    return data;
  }

  // Get data for charts
  Future<Map<String, dynamic>> getChartData() async {
    final data = _userData ?? await _getUserData();
    return {
      'days': data['days'],
      'stage': data['stage'],
      'hasSupport': data['hasSupport'],
    };
  }
}

class RecoveryProgressScreen extends StatelessWidget {
  const RecoveryProgressScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => RecoveryAnalyticsProvider(
        apiKey: 'AIzaSyCOutG-g_tVZKzbTtH0bzNjWdoaDVA2YCo',
      ),
      child: Scaffold(
        appBar: AppBar(
          title: Text('Recovery Analytics',
              style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w600, fontSize: 20)),
          centerTitle: true,
          actions: [
            Consumer<RecoveryAnalyticsProvider>(
              builder: (context, provider, _) {
                return IconButton(
                  icon: provider.isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.refresh),
                  onPressed: provider.isLoading
                      ? null
                      : () {
                          provider.refreshAnalysis();
                        },
                  tooltip: 'Refresh Analysis',
                );
              },
            ),
          ],
        ),
        body: const SafeArea(
          child: _ProgressContentView(),
        ),
      ),
    );
  }
}

class _ProgressContentView extends StatefulWidget {
  const _ProgressContentView();

  @override
  State<_ProgressContentView> createState() => _ProgressContentViewState();
}

class _ProgressContentViewState extends State<_ProgressContentView> {
  int _selectedViewIndex = 0;
  final List<String> _viewOptions = ['Summary', 'Visual', 'Detailed'];

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // View selector
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: SegmentedButton<int>(
            segments: _viewOptions
                .asMap()
                .entries
                .map((entry) => ButtonSegment<int>(
                      value: entry.key,
                      label: Text(entry.value),
                    ))
                .toList(),
            selected: {_selectedViewIndex},
            onSelectionChanged: (Set<int> newSelection) {
              setState(() {
                _selectedViewIndex = newSelection.first;
              });
            },
          ),
        ),
        // Content based on selection
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: IndexedStack(
              index: _selectedViewIndex,
              children: [
                _buildSummaryView(),
                _buildVisualView(),
                _buildDetailedView(),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryView() {
    return SingleChildScrollView(
      child: Column(
        children: [
          _buildProgressCard(),
          const SizedBox(height: 16),
          Consumer<RecoveryAnalyticsProvider>(
            builder: (context, provider, _) {
              return _buildAnalysisCard(provider);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildVisualView() {
    return Consumer<RecoveryAnalyticsProvider>(
      builder: (context, provider, _) {
        return FutureBuilder<Map<String, dynamic>>(
          future: provider.getChartData(),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return _buildShimmerLoader();
            }

            if (!snapshot.hasData || snapshot.hasError) {
              return Center(
                child: Text(
                  'Unable to load chart data',
                  style: GoogleFonts.poppins(color: Colors.grey.shade700),
                ),
              );
            }

            final data = snapshot.data!;
            return SingleChildScrollView(
              child: Column(
                children: [
                  _buildDaysChart(data['days']),
                  const SizedBox(height: 16),
                  _buildStagePieChart(data['stage']),
                  const SizedBox(height: 16),
                  _buildSupportChart(data['hasSupport']),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildDetailedView() {
    return SingleChildScrollView(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        children: [
          Consumer<RecoveryAnalyticsProvider>(
            builder: (context, provider, _) {
              return _buildStatsGrid(provider);
            },
          ),
          const SizedBox(height: 16),
          Consumer<RecoveryAnalyticsProvider>(
            builder: (context, provider, _) {
              return _buildEnhancedAnalysisCard(provider);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildProgressCard() {
    return Consumer<RecoveryAnalyticsProvider>(
      builder: (context, provider, _) {
        return Card(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: FutureBuilder<Map<String, dynamic>>(
              future: provider.getChartData(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(
                    child: SizedBox(
                      height: 100,
                      child: CircularProgressIndicator(),
                    ),
                  );
                }

                if (!snapshot.hasData || snapshot.hasError) {
                  return const Center(
                    child: Text('Failed to load progress data'),
                  );
                }

                final data = snapshot.data!;
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Recovery Progress',
                            style: GoogleFonts.poppins(
                                fontSize: 18, fontWeight: FontWeight.w600)),
                        Chip(
                          label: Text('Day ${data['days']}',
                              style: GoogleFonts.poppins(color: Colors.white)),
                          backgroundColor: Colors.blue.shade700,
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    LinearProgressIndicator(
                      value: _calculateProgressValue(data['stage']),
                      minHeight: 12,
                      borderRadius: BorderRadius.circular(6),
                      backgroundColor: Colors.grey.shade200,
                      valueColor: AlwaysStoppedAnimation<Color>(
                          _getStageColor(data['stage'])),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _buildStageIndicator('Beginning', data['stage']),
                        _buildStageIndicator('Middle', data['stage']),
                        _buildStageIndicator('Late', data['stage']),
                        _buildStageIndicator('Maintenance', data['stage']),
                      ],
                    ),
                  ],
                );
              },
            ),
          ),
        );
      },
    );
  }

  double _calculateProgressValue(String stage) {
    switch (stage.toLowerCase()) {
      case 'beginning':
        return 0.25;
      case 'middle':
        return 0.5;
      case 'late':
        return 0.75;
      case 'maintenance':
        return 1.0;
      default:
        return 0.1;
    }
  }

  Color _getStageColor(String stage) {
    switch (stage.toLowerCase()) {
      case 'beginning':
        return Colors.orange;
      case 'middle':
        return Colors.blue;
      case 'late':
        return Colors.green;
      case 'maintenance':
        return Colors.purple;
      default:
        return Colors.blueGrey;
    }
  }

  Widget _buildStageIndicator(String label, String currentStage) {
    final isActive = label.toLowerCase() == currentStage.toLowerCase();
    return Column(
      children: [
        Icon(
          isActive ? Icons.location_on : Icons.location_on_outlined,
          color: isActive ? _getStageColor(currentStage) : Colors.grey,
          size: 20,
        ),
        Text(label,
            style: GoogleFonts.poppins(
                fontSize: 12,
                color: isActive ? _getStageColor(currentStage) : Colors.grey)),
      ],
    );
  }

  Widget _buildAnalysisCard(RecoveryAnalyticsProvider provider) {
    if (provider.isLoading) {
      return _buildShimmerLoader();
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('AI Analysis',
                    style: GoogleFonts.poppins(
                        fontSize: 18, fontWeight: FontWeight.w600)),
                if (provider.lastUpdated != null)
                  Tooltip(
                    message: 'Last updated',
                    child: Text(
                      DateFormat('MMM d, h:mm a').format(provider.lastUpdated!),
                      style: GoogleFonts.poppins(
                          fontSize: 12, color: Colors.grey.shade600),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            // Using a Container with a fixed height prevents layout issues
            Container(
              constraints: const BoxConstraints(minHeight: 150),
              child: provider.analysis.isEmpty
                  ? const Center(child: Text('No analysis available'))
                  : MarkdownBody(
                      data: provider.analysis,
                      styleSheet: MarkdownStyleSheet(
                        p: GoogleFonts.poppins(fontSize: 14, height: 1.6),
                        strong: GoogleFonts.poppins(
                            fontWeight: FontWeight.w600,
                            color: Colors.blue.shade800),
                        listBullet: GoogleFonts.poppins(fontSize: 14),
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDaysChart(int days) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Recovery Journey',
                style: GoogleFonts.poppins(
                    fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            SizedBox(
              height: 200,
              child: SfCartesianChart(
                primaryXAxis: CategoryAxis(isVisible: false),
                primaryYAxis: NumericAxis(isVisible: false),
                plotAreaBorderWidth: 0,
                series: <CartesianSeries>[
                  ColumnSeries<_ChartData, String>(
                    dataSource: [_ChartData('', days)],
                    xValueMapper: (_ChartData data, _) => data.x,
                    yValueMapper: (_ChartData data, _) => data.y,
                    color: Colors.blue.shade400,
                    width: 0.5,
                    borderRadius: BorderRadius.circular(4),
                  )
                ],
                annotations: <CartesianChartAnnotation>[
                  CartesianChartAnnotation(
                    widget: Text('$days days',
                        style: GoogleFonts.poppins(
                            fontSize: 24, fontWeight: FontWeight.bold)),
                    coordinateUnit: CoordinateUnit.point,
                    x: '',
                    y: days.toDouble(),
                    horizontalAlignment: ChartAlignment.center,
                    verticalAlignment: ChartAlignment.far,
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStagePieChart(String stage) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Current Stage',
                style: GoogleFonts.poppins(
                    fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            SizedBox(
              height: 200,
              child: SfCircularChart(
                legend: Legend(
                    isVisible: true,
                    overflowMode: LegendItemOverflowMode.wrap,
                    textStyle: GoogleFonts.poppins(fontSize: 12)),
                series: <CircularSeries>[
                  DoughnutSeries<_StageData, String>(
                    dataSource: [
                      _StageData('Current Stage', 1),
                      _StageData('Remaining Journey', 3),
                    ],
                    xValueMapper: (_StageData data, _) => data.x,
                    yValueMapper: (_StageData data, _) => data.y,
                    pointColorMapper: (_StageData data, _) {
                      return data.x == 'Current Stage'
                          ? _getStageColor(stage)
                          : Colors.grey.shade200;
                    },
                    innerRadius: '70%',
                    dataLabelSettings:
                        const DataLabelSettings(isVisible: false),
                  ),
                ],
                annotations: <CircularChartAnnotation>[
                  CircularChartAnnotation(
                    widget: Text(stage,
                        style: GoogleFonts.poppins(
                            fontSize: 16, fontWeight: FontWeight.bold)),
                    angle: 0,
                    radius: '0%',
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSupportChart(bool hasSupport) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Support System',
                style: GoogleFonts.poppins(
                    fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            SizedBox(
              height: 150,
              child: SfCartesianChart(
                primaryXAxis: CategoryAxis(isVisible: false),
                primaryYAxis: NumericAxis(isVisible: false),
                plotAreaBorderWidth: 0,
                series: <CartesianSeries>[
                  BarSeries<_SupportData, String>(
                    dataSource: [_SupportData('', hasSupport ? 1.0 : 0.0)],
                    xValueMapper: (_SupportData data, _) => data.x,
                    yValueMapper: (_SupportData data, _) => data.y,
                    color: hasSupport
                        ? Colors.green.shade400
                        : Colors.red.shade400,
                    width: 0.3,
                    borderRadius: BorderRadius.circular(4),
                  )
                ],
                annotations: <CartesianChartAnnotation>[
                  CartesianChartAnnotation(
                    widget: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          hasSupport ? Icons.group : Icons.person,
                          size: 40,
                          color: hasSupport ? Colors.green : Colors.red,
                        ),
                        Text(
                          hasSupport ? 'Supported' : 'No Support',
                          style: GoogleFonts.poppins(
                              fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                    coordinateUnit: CoordinateUnit.point,
                    x: '',
                    y: hasSupport ? 1.0 : 0.0,
                    horizontalAlignment: ChartAlignment.center,
                    verticalAlignment: ChartAlignment.center,
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatsGrid(RecoveryAnalyticsProvider provider) {
    return FutureBuilder<Map<String, dynamic>>(
      future: provider.getChartData(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return _buildShimmerLoader();
        }

        if (!snapshot.hasData || snapshot.hasError) {
          return const Center(
            child: Text('Failed to load statistics'),
          );
        }

        final data = snapshot.data!;
        return GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          childAspectRatio: 1.5,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
          padding: const EdgeInsets.all(8),
          children: [
            _buildStatTile('Total Days', data['days'].toString(),
                Icons.calendar_today, Colors.blue),
            _buildStatTile('Recovery Stage', data['stage'], Icons.timeline,
                _getStageColor(data['stage'])),
            _buildStatTile(
                'Addiction Type',
                data.containsKey('type') ? data['type'] : 'General',
                Icons.medical_services,
                Colors.purple),
            _buildStatTile('Support System', data['hasSupport'] ? 'Yes' : 'No',
                Icons.group, data['hasSupport'] ? Colors.green : Colors.red),
          ],
        );
      },
    );
  }

  Widget _buildStatTile(
      String title, String value, IconData icon, Color color) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          mainAxisSize: MainAxisSize.min, // Add this line
          children: [
            Icon(icon, size: 28, color: color), // Reduced icon size
            const SizedBox(height: 4), // Reduced spacing
            Text(title,
                textAlign: TextAlign.center,
                style: GoogleFonts.poppins(
                    fontSize: 12,
                    color: Colors.grey.shade600)), // Reduced font size
            const SizedBox(height: 2), // Reduced spacing
            Text(value,
                textAlign: TextAlign.center,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: color)), // Reduced font size
          ],
        ),
      ),
    );
  }

  Widget _buildEnhancedAnalysisCard(RecoveryAnalyticsProvider provider) {
    if (provider.isLoading) {
      return _buildShimmerLoader();
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Detailed Analysis',
                    style: GoogleFonts.poppins(
                        fontSize: 18, fontWeight: FontWeight.w600)),
                if (provider.lastUpdated != null)
                  Chip(
                    label: Text(
                        'Updated ${DateFormat('MMM d').format(provider.lastUpdated!)}',
                        style: GoogleFonts.poppins(fontSize: 12)),
                    backgroundColor: Colors.grey.shade200,
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (provider.analysis.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade50,
                  borderRadius: BorderRadius.circular(12),
                ),
                // Set a fixed height to avoid layout issues
                constraints: const BoxConstraints(minHeight: 200),
                child: MarkdownBody(
                  data: provider.analysis,
                  styleSheet: MarkdownStyleSheet(
                    p: GoogleFonts.poppins(fontSize: 14, height: 1.6),
                    h1: GoogleFonts.poppins(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue.shade800),
                    h2: GoogleFonts.poppins(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue.shade700),
                    strong: GoogleFonts.poppins(
                        fontWeight: FontWeight.w600,
                        color: Colors.blue.shade800),
                    listBullet: GoogleFonts.poppins(fontSize: 14),
                    blockquote: GoogleFonts.poppins(
                        fontSize: 14,
                        fontStyle: FontStyle.italic,
                        color: Colors.grey.shade700),
                    blockquoteDecoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      border: Border(
                        left: BorderSide(color: Colors.blue.shade300, width: 4),
                      ),
                    ),
                  ),
                ),
              ),
            const SizedBox(height: 16),
            _buildActionButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Consumer<RecoveryAnalyticsProvider>(
      builder: (context, provider, _) {
        return Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                icon: const Icon(Icons.share, size: 18),
                label: Text('Share', style: GoogleFonts.poppins()),
                onPressed: provider.isLoading
                    ? null
                    : () {
                        // Implement share functionality
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                              content:
                                  Text('Share functionality not implemented')),
                        );
                      },
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: FilledButton.icon(
                icon: provider.isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.white))
                    : const Icon(Icons.refresh, size: 18),
                label: Text(provider.isLoading ? 'Refreshing...' : 'Refresh',
                    style: GoogleFonts.poppins()),
                onPressed: provider.isLoading
                    ? null
                    : () {
                        provider.refreshAnalysis();
                      },
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildShimmerLoader() {
    return Shimmer.fromColors(
      baseColor: Colors.grey.shade300,
      highlightColor: Colors.grey.shade100,
      child: Column(
        children: [
          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: const SizedBox(height: 200, width: double.infinity),
          ),
          const SizedBox(height: 16),
          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: const SizedBox(height: 150, width: double.infinity),
          ),
        ],
      ),
    );
  }
}

// Data classes for charts
class _ChartData {
  final String x;
  final int y;
  _ChartData(this.x, this.y);
}

class _StageData {
  final String x;
  final int y;
  _StageData(this.x, this.y);
}

class _SupportData {
  final String x;
  final double y;
  _SupportData(this.x, this.y);
}
